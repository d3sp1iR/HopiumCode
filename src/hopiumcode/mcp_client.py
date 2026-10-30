"""Клиент MCP для оркестратора (зона участников 1 и 4).

Запускает MCP-сервер подпроцессом и общается с ним по stdio: JSON-RPC 2.0,
одно сообщение на строку. Протокол реализован напрямую, без клиентской части
SDK: оркестратору нужен синхронный интерфейс, а клиентский API SDK асинхронный
и между mcp 1.x и 2.x заметно поменялся. Сервер (FastMCP / MCPServer)
понимает рукопожатие 2025 года в обеих версиях SDK.

    with McpClient(env={"REPO_ROOT": "data/repo/flask/src/flask"}) as client:
        tools = client.list_tools_for_ollama()       # -> ollama.chat(tools=tools)
        text = client.call_tool("read_file", {"path": "app.py", "end_line": 20})

Для оркестратора есть и модульные функции (ТЗ, П1.2: «mcp_client.call_tool(name, args)»):
общий клиент запускается при первом вызове и закрывается при выходе из программы.

    from hopiumcode import mcp_client
    tools = mcp_client.list_tools_for_ollama()
    text = mcp_client.call_tool("git_log", {"path": "app.py"})
"""

import atexit
import collections
import itertools
import json
import logging
import os
import queue
import subprocess
import sys
import threading
from typing import Any

log = logging.getLogger("hopiumcode.mcp_client")

PROTOCOL_VERSION = "2025-06-18"
CALL_TIMEOUT = 20.0      # секунд на любой запрос к серверу
CLOSE_TIMEOUT = 3.0      # сколько ждать мирного завершения сервера
STDERR_TAIL = 20         # сколько последних строк stderr сервера помнить для ошибок

_CLOSED = object()       # сигнал «сервер закрыл stdout» для ожидающих ответов


class McpClientError(RuntimeError):
    """Сбой транспорта или протокола (сервер не запустился, умер, молчит)."""


class McpClient:
    def __init__(
        self,
        command: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout: float = CALL_TIMEOUT,
    ) -> None:
        self.command = command or [sys.executable, "-m", "hopiumcode.mcp_server.server"]
        self.env = {
            **os.environ,
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
            **(env or {}),
        }
        self.cwd = cwd
        self.timeout = timeout

        self._proc: subprocess.Popen[str] | None = None
        self._ids = itertools.count(1)
        self._pending: dict[int, queue.Queue] = {}
        self._pending_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._stderr_tail: collections.deque[str] = collections.deque(maxlen=STDERR_TAIL)
        self._threads: list[threading.Thread] = []
        self._stdout_eof = threading.Event()  # сервер закрыл stdout (умер или выходит)
        self._schemas: dict[str, dict] = {}  # inputSchema по имени инструмента
        self._closed = False                 # close() вызван явно: сам не перезапускаемся
        self._atexit_registered = False
        self.server_info: dict = {}

    # --- жизненный цикл -----------------------------------------------------
    def start(self) -> None:
        """Запустить сервер и выполнить рукопожатие initialize.

        Бросает McpClientError, если сервер не поднялся: без инструментов
        оркестратору честнее упасть сразу (или перейти на заглушку).
        """
        if self.is_running:
            return
        self._closed = False
        if not self._atexit_registered:  # оркестратор может не вызвать close()
            atexit.register(self.close)
            self._atexit_registered = True
        try:
            self._proc = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self.env,
                cwd=self.cwd,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as exc:
            raise McpClientError(f"не удалось запустить MCP-сервер {self.command}: {exc}") from exc

        self._stdout_eof = threading.Event()
        self._threads = [
            threading.Thread(target=self._read_stdout, name="mcp-stdout", daemon=True),
            threading.Thread(target=self._read_stderr, name="mcp-stderr", daemon=True),
        ]
        for t in self._threads:
            t.start()

        try:
            result = self._request("initialize", {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "hopiumcode", "version": "0.1.0"},
            })
        except (McpClientError, TimeoutError) as exc:
            hint = self._stderr_hint()
            self.close()
            if isinstance(exc, TimeoutError):
                raise McpClientError(
                    f"MCP-сервер не ответил на initialize за {self.timeout:g} с{hint}"
                ) from None
            raise
        self.server_info = result.get("serverInfo", {})
        self._notify("notifications/initialized")
        log.info("MCP-сервер запущен: %s, протокол %s",
                 self.server_info.get("name"), result.get("protocolVersion"))

    def close(self) -> None:
        self._closed = True
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()  # EOF на stdin — штатный сигнал серверу завершиться
            proc.wait(timeout=CLOSE_TIMEOUT)
        except (OSError, subprocess.TimeoutExpired):
            proc.terminate()
            try:
                proc.wait(timeout=CLOSE_TIMEOUT)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        for t in self._threads:
            t.join(timeout=1)
        self._fail_pending()

    @property
    def is_running(self) -> bool:
        return (self._proc is not None and self._proc.poll() is None
                and not self._stdout_eof.is_set())

    def __enter__(self) -> "McpClient":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- публичный API для оркестратора ---------------------------------------
    def _ensure_started(self) -> None:
        """Ленивый запуск: оркестратор создаёт McpClient() и сразу зовёт методы, без start()."""
        if self._proc is None and not self._closed:
            self.start()

    def list_tools_for_ollama(self) -> list[dict]:
        """Список инструментов в формате параметра tools пакета ollama.

        Если сервер ещё не запущен — запускает его; не поднялся — McpClientError.
        """
        self._ensure_started()
        tools: list[dict] = []
        cursor = None
        while True:
            result = self._request("tools/list", {"cursor": cursor} if cursor else {})
            for tool in result.get("tools", []):
                schema = tool.get("inputSchema") or {}
                self._schemas[tool["name"]] = schema
                tools.append(mcp_tool_to_ollama(tool))
            cursor = result.get("nextCursor")
            if not cursor:
                return tools

    def call_tool(self, name: str, args: dict | str | None = None) -> str:
        """Вызвать инструмент. Никогда не бросает: любая ошибка — текст «Ошибка: …»."""
        try:
            self._ensure_started()
        except McpClientError as exc:
            return f"Ошибка: {exc}"
        if not self.is_running:
            return f"Ошибка: MCP-сервер не запущен{self._stderr_hint()}"
        try:
            arguments = self._prepare_args(name, args)
        except ValueError as exc:
            return f"Ошибка: {exc}"
        try:
            result = self._request("tools/call", {"name": name, "arguments": arguments})
        except TimeoutError:
            # Сервер мог зависнуть на этом вызове: перезапускаем, чтобы не блокировать следующие
            self._restart()
            return f"Ошибка: инструмент {name} не ответил за {self.timeout:g} с"
        except McpRpcError as exc:
            return f"Ошибка: {exc}"
        except McpClientError as exc:
            return f"Ошибка: {exc}"
        return _result_to_text(name, result)

    # --- подготовка аргументов --------------------------------------------------
    def _prepare_args(self, name: str, args: dict | str | None) -> dict:
        """Мелкие модели путаются в типах: чиним то, что можно починить однозначно."""
        if args is None:
            return {}
        if isinstance(args, str):  # некоторые модели отдают arguments строкой JSON
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError as exc:
                raise ValueError(f"аргументы не являются JSON: {exc.msg}") from exc
        if not isinstance(args, dict):
            raise ValueError("аргументы должны быть объектом {имя: значение}")

        schema = self._schemas.get(name)
        if not schema:
            return args
        props = schema.get("properties", {})
        fixed = {}
        for key, value in args.items():
            if key not in props:
                log.info("%s: лишний аргумент %r отброшен", name, key)
                continue
            if value is None:  # null для необязательного аргумента = «не указан»
                continue
            fixed[key] = _coerce(value, _plain_type(props[key]))
        return fixed

    # --- JSON-RPC ---------------------------------------------------------------
    def _request(self, method: str, params: dict) -> dict:
        if not self.is_running:
            raise McpClientError(f"MCP-сервер не запущен{self._stderr_hint()}")
        req_id = next(self._ids)
        box: queue.Queue = queue.Queue(maxsize=1)
        with self._pending_lock:
            self._pending[req_id] = box
        try:
            self._send({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
            try:
                msg = box.get(timeout=self.timeout)
            except queue.Empty:
                self._send_cancel(req_id)
                raise TimeoutError(method) from None
        finally:
            with self._pending_lock:
                self._pending.pop(req_id, None)

        if msg is _CLOSED:
            self._drain_stderr()
            raise McpClientError(f"MCP-сервер завершился{self._stderr_hint()}")
        if "error" in msg:
            err = msg["error"] or {}
            raise McpRpcError(err.get("message", "неизвестная ошибка"), err.get("code"))
        return msg.get("result") or {}

    def _notify(self, method: str, params: dict | None = None) -> None:
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        self._send(msg)

    def _send_cancel(self, req_id: int) -> None:
        try:
            self._notify("notifications/cancelled", {"requestId": req_id, "reason": "timeout"})
        except McpClientError:
            pass

    def _send(self, msg: dict) -> None:
        line = json.dumps(msg, ensure_ascii=False)  # без \n внутри: одна строка = одно сообщение
        with self._write_lock:
            proc = self._proc
            if proc is None or proc.stdin is None:
                raise McpClientError("MCP-сервер не запущен")
            try:
                proc.stdin.write(line + "\n")
                proc.stdin.flush()
            except (BrokenPipeError, OSError, ValueError) as exc:
                raise McpClientError(f"MCP-сервер закрыл stdin{self._stderr_hint()}") from exc

    def _read_stdout(self) -> None:
        proc, eof = self._proc, self._stdout_eof
        assert proc is not None and proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                log.warning("не-JSON в stdout сервера (print в коде сервера?): %.200s", line)
                continue
            if "method" in msg:  # запрос или уведомление от сервера
                self._handle_server_message(msg)
                continue
            with self._pending_lock:
                box = self._pending.get(msg.get("id"))
            if box is not None:
                box.put(msg)
            # ответ на уже отменённый запрос просто отбрасываем
        eof.set()
        self._fail_pending()

    def _handle_server_message(self, msg: dict) -> None:
        if "id" not in msg:  # уведомление: логи, прогресс — нам не нужны
            log.debug("уведомление сервера: %s", msg.get("method"))
            return
        if msg["method"] == "ping":
            reply = {"jsonrpc": "2.0", "id": msg["id"], "result": {}}
        else:  # sampling, roots, elicitation — клиент их не поддерживает
            reply = {"jsonrpc": "2.0", "id": msg["id"],
                     "error": {"code": -32601, "message": f"не поддерживается: {msg['method']}"}}
        try:
            self._send(reply)
        except McpClientError:
            pass

    def _read_stderr(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stderr is not None
        for line in proc.stderr:
            line = line.rstrip()
            if line:
                self._stderr_tail.append(line)
                log.debug("[server] %s", line)

    def _fail_pending(self) -> None:
        with self._pending_lock:
            boxes = list(self._pending.values())
        for box in boxes:
            try:
                box.put_nowait(_CLOSED)
            except queue.Full:
                pass

    def _restart(self) -> None:
        log.warning("MCP-сервер не ответил за %s с, перезапуск", self.timeout)
        self.close()
        try:
            self.start()
            if self._schemas:
                self.list_tools_for_ollama()  # восстановить кэш схем
        except McpClientError as exc:
            log.error("перезапуск не удался: %s", exc)

    def _drain_stderr(self) -> None:
        """Дождаться последних строк stderr умершего сервера — в них причина."""
        for t in self._threads:
            if t.name == "mcp-stderr":
                t.join(timeout=1)

    def _stderr_hint(self) -> str:
        return f": {self._stderr_tail[-1]}" if self._stderr_tail else ""


class McpRpcError(McpClientError):
    def __init__(self, message: str, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code


# --- конвертация схем MCP -> ollama -------------------------------------------
_KEEP = ("type", "description", "enum", "items", "properties", "required")


def mcp_tool_to_ollama(tool: dict) -> dict:
    """{"name", "description", "inputSchema"} -> {"type": "function", "function": {...}}."""
    schema = tool.get("inputSchema") or {}
    params = simplify_schema(schema, schema.get("$defs") or schema.get("definitions") or {})
    params["type"] = "object"
    params.setdefault("properties", {})
    params["required"] = list(schema.get("required", []))
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": (tool.get("description") or "").strip(),
            "parameters": params,
        },
    }


def simplify_schema(node: Any, defs: dict) -> Any:
    """Упрощает JSON Schema от pydantic до того, что понимают ollama и маленькие модели.

    - $ref раскрывается из $defs;
    - anyOf [X, null] (так pydantic описывает `int | None`) превращается в X;
    - title, default, additionalProperties и прочее выбрасываются,
      а значение по умолчанию дописывается в description словами.
    """
    if not isinstance(node, dict):
        return node
    if "$ref" in node:
        target = defs.get(node["$ref"].rsplit("/", 1)[-1], {})
        return simplify_schema({**target, **{k: v for k, v in node.items() if k != "$ref"}}, defs)

    variants = node.get("anyOf") or node.get("oneOf")
    if variants:
        non_null = [v for v in variants if not (isinstance(v, dict) and v.get("type") == "null")]
        base = simplify_schema(non_null[0], defs) if len(non_null) == 1 else {"type": "string"}
        merged = {**node, **base}
        merged.pop("anyOf", None)
        merged.pop("oneOf", None)
        if "description" in node:
            merged["description"] = node["description"]
        node = merged

    out = {k: node[k] for k in _KEEP if k in node}
    if isinstance(out.get("type"), list):  # ["integer", "null"] -> "integer"
        types = [t for t in out["type"] if t != "null"]
        out["type"] = types[0] if len(types) == 1 else "string"
    if "properties" in out:
        out["properties"] = {k: simplify_schema(v, defs) for k, v in out["properties"].items()}
    if "items" in out:
        out["items"] = simplify_schema(out["items"], defs)
    default = node.get("default")
    if default is not None and "по умолчанию" not in out.get("description", "").lower():
        out["description"] = (out.get("description", "") + f" По умолчанию {default}.").strip()
    return out


def _plain_type(prop: dict) -> str | None:
    return simplify_schema(prop, {}).get("type")


def _coerce(value: Any, expected: str | None) -> Any:
    """"42" -> 42 для integer и т. п.; всё неоднозначное оставляем серверу на проверку."""
    if isinstance(value, str):
        text = value.strip()
        if expected == "integer" and text.lstrip("-").isdigit():
            return int(text)
        if expected == "number":
            try:
                return float(text)
            except ValueError:
                return value
        if expected == "boolean" and text.lower() in ("true", "false"):
            return text.lower() == "true"
    if expected == "integer" and isinstance(value, float) and value.is_integer():
        return int(value)
    if expected == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return value


def _result_to_text(name: str, result: dict) -> str:
    parts = []
    for item in result.get("content") or []:
        if item.get("type") == "text":
            parts.append(item.get("text", ""))
        else:
            parts.append(f"[{item.get('type', 'содержимое')} не показан]")
    # Только переводы строк по краям: ведущий пробел значим — это формат " 12|код"
    text = "\n".join(parts).strip("\n")
    if not text.strip() and result.get("structuredContent") is not None:
        text = json.dumps(result["structuredContent"], ensure_ascii=False)
    if result.get("isError"):
        # FastMCP 1.x оборачивает исключение: "Error executing tool read_file: <текст>"
        prefix = f"Error executing tool {name}: "
        text = text.strip().removeprefix(prefix)
        if not text or text == prefix.rstrip(": "):  # SDK скрыл текст исключения
            text = f"инструмент {name} завершился с ошибкой (подробности в stderr сервера)"
        return f"Ошибка: {text}"
    return text if text.strip() else "(пустой результат)"


# --- модульный интерфейс для оркестратора ----------------------------------------
_default_client: McpClient | None = None
_default_lock = threading.Lock()


def get_client() -> McpClient:
    """Общий клиент процесса; если сервер упал — поднимается заново."""
    global _default_client
    with _default_lock:
        if _default_client is None or not _default_client.is_running:
            if _default_client is not None:
                _default_client.close()
            client = McpClient()
            client.start()  # McpClientError, если сервер не поднялся
            atexit.register(client.close)
            _default_client = client
        return _default_client


def list_tools_for_ollama() -> list[dict]:
    return get_client().list_tools_for_ollama()


def call_tool(name: str, args: dict | str | None = None) -> str:
    """Как McpClient.call_tool: никогда не бросает, ошибки — текстом."""
    try:
        client = get_client()
        if not client._schemas:  # схемы нужны для приведения типов аргументов
            client.list_tools_for_ollama()
    except (McpClientError, TimeoutError) as exc:
        return f"Ошибка: {exc}"
    return client.call_tool(name, args)


def close() -> None:
    global _default_client
    with _default_lock:
        if _default_client is not None:
            _default_client.close()
            _default_client = None
