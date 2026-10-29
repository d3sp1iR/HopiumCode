"""MCP-клиент для цикла оркестратора. Владельцы: участники 4 и 1.

McpClient реализует interfaces.ToolClientLike: запускает MCP-сервер (python -m hopiumcode.mcp_server.server)
подпроцессом по stdio при первом обращении, отдаёт схемы инструментов в формате tools пакета ollama и
вызывает инструменты. Асинхронная сессия MCP живёт в отдельном потоке со своим event loop, синхронные методы
передают в него вызовы. Ошибки (сервер не запустился, таймаут, сбой инструмента) возвращаются текстом
«Ошибка: ...», исключений наружу нет: агент читает причину и продолжает.
"""
from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import os
import sys
import threading
from pathlib import Path
from typing import Self

TIMEOUT = 20.0
_SCHEMA_KEYS = ("type", "description", "enum", "items", "default")


def _errlog():
    """stderr для подпроцесса: нужен настоящий файл (под pytest sys.stderr без fileno)."""
    try:
        sys.stderr.fileno()
        return sys.stderr
    except (AttributeError, OSError, ValueError):
        return sys.__stderr__


def _simplify(prop: dict) -> dict:
    """JSON Schema аргумента в простой вид для маленьких моделей: int | None -> integer, без title."""
    prop = dict(prop)
    if "anyOf" in prop and "type" not in prop:
        types = [p for p in prop.pop("anyOf") if p.get("type") != "null"]
        if len(types) == 1:
            prop = {**types[0], **prop}
    return {k: v for k, v in prop.items() if k in _SCHEMA_KEYS}


def to_ollama_tool(name: str, description: str | None, input_schema: dict) -> dict:
    props = {k: _simplify(v) for k, v in (input_schema.get("properties") or {}).items()}
    params = {"type": "object", "properties": props}
    if input_schema.get("required"):
        params["required"] = list(input_schema["required"])
    return {"type": "function", "function": {"name": name, "description": description or "", "parameters": params}}


class McpClient:
    def __init__(self, root: Path | None = None, timeout: float = TIMEOUT, command: list[str] | None = None) -> None:
        self.root = Path(root) if root else None
        self.timeout = timeout
        self.command = command or [sys.executable, "-m", "hopiumcode.mcp_server.server"]
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._serving: concurrent.futures.Future | None = None
        self._session = None
        self._stop: asyncio.Event | None = None
        self._start_error: str | None = None
        self._tools_cache: list[dict] | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ жизненный цикл

    def _params(self):
        from mcp.client.stdio import StdioServerParameters

        from .mcp_server.server import get_root
        package_parent = str(Path(__file__).resolve().parents[1])
        env = {"REPO_ROOT": str((self.root or get_root()).resolve()),
               "PYTHONPATH": os.pathsep.join(filter(None, [package_parent, os.environ.get("PYTHONPATH")]))}
        return StdioServerParameters(command=self.command[0], args=self.command[1:], env=env)

    async def _serve(self, ready: concurrent.futures.Future) -> None:
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client
        try:
            async with stdio_client(self._params(), errlog=_errlog()) as (read, write), \
                    ClientSession(read, write) as session:
                await session.initialize()
                self._session = session
                self._stop = asyncio.Event()
                ready.set_result(True)
                await self._stop.wait()
        except BaseException as ex:  # noqa: BLE001 - причина отдаётся в start() текстом
            if not ready.done():
                ready.set_exception(ex)
        finally:
            self._session = None

    def start(self) -> None:
        """Запускает сервер (повторный вызов ничего не делает). Не бросает исключений."""
        with self._lock:
            if self._session is not None or self._start_error is not None:
                return
            if self._loop is not None:  # сессия оборвалась (сервер упал): убираем старый поток и запускаем заново
                self._shutdown_loop()
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._loop.run_forever, name="mcp-client", daemon=True)
            self._thread.start()
            ready: concurrent.futures.Future = concurrent.futures.Future()
            self._serving = asyncio.run_coroutine_threadsafe(self._serve(ready), self._loop)
            try:
                ready.result(timeout=self.timeout)
            except concurrent.futures.TimeoutError:
                self._start_error = f"MCP-сервер не ответил за {self.timeout:.0f} c"
            except Exception as ex:  # noqa: BLE001
                self._start_error = f"MCP-сервер не запустился: {ex!r}"
            if self._start_error:
                print(f"⚠ {self._start_error}", file=sys.stderr)
                self._shutdown_loop()
            else:
                atexit.register(self.close)

    def close(self) -> None:
        """Останавливает сессию и подпроцесс сервера. Повторный вызов безопасен."""
        with self._lock:
            loop, stop = self._loop, self._stop
            if loop is None:
                return
            if stop is not None and loop.is_running():
                loop.call_soon_threadsafe(stop.set)
                try:
                    self._serving.result(timeout=10)
                except Exception:  # noqa: BLE001, S110 - закрываемся в любом случае
                    pass
            self._shutdown_loop()
            self._stop = None
            self._tools_cache = None
            self._start_error = None

    def _shutdown_loop(self) -> None:
        loop, thread = self._loop, self._thread
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
            if thread is not None:
                thread.join(timeout=5)
            if not loop.is_running():
                loop.close()
        self._loop = self._thread = None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ вызовы

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=self.timeout + 1)

    def list_tools_for_ollama(self) -> list[dict]:
        self.start()
        if self._session is None:
            return []
        if self._tools_cache is None:
            try:
                result = self._run(self._session.list_tools())
            except Exception as ex:  # noqa: BLE001
                print(f"⚠ MCP: список инструментов не получен: {ex!r}", file=sys.stderr)
                return []
            self._tools_cache = [to_ollama_tool(t.name, t.description, t.input_schema) for t in result.tools]
        return [dict(t) for t in self._tools_cache]

    def call_tool(self, name: str, args: dict) -> str:
        self.start()
        if self._session is None:
            return f"Ошибка: {self._start_error or 'MCP-сервер не запущен'}"
        try:
            result = self._run(self._session.call_tool(name, dict(args or {}), read_timeout_seconds=self.timeout))
        except (concurrent.futures.TimeoutError, TimeoutError):
            return f"Ошибка: инструмент {name} не ответил за {self.timeout:.0f} c"
        except Exception as ex:  # noqa: BLE001 - любая ошибка транспорта возвращается текстом
            if "timed out" in str(ex).lower():  # таймаут запроса в SDK приходит как ошибка MCP
                return f"Ошибка: инструмент {name} не ответил за {self.timeout:.0f} c"
            return f"Ошибка: вызов {name} не удался: {ex}"
        text = "\n".join(getattr(c, "text", "") for c in result.content if getattr(c, "text", None))
        if result.is_error and not text.startswith("Ошибка"):
            text = f"Ошибка: {text or 'инструмент ' + name + ' завершился с ошибкой'}"
        return text
