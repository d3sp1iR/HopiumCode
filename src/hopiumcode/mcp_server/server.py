"""MCP-сервер, транспорт stdio, инструменты только на чтение. Владелец: участник 4.

Инструменты: read_file, search_code, find_callers, git_log, git_blame.
Безопасность: пути только внутри REPO_ROOT, лимит ответа 8000 символов, git без shell.

Запуск: python -m hopiumcode.mcp_server.server (корень: переменная REPO_ROOT, иначе Config.repo_root).
SDK: пакет mcp 2.x, класс MCPServer (в mcp 1.x он назывался FastMCP).
Каждый инструмент — обычная функция с явным root: её можно тестировать без MCP. Ошибки возвращаются
текстом «Ошибка: ...», а не исключением, чтобы модель могла прочитать причину.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from pydantic import Field

SERVER_NAME = "hopiumcode"
MAX_CHARS = 8000
DEFAULT_LINES = 200
MAX_LINES = 400
READ_SUFFIXES = {".py"}  # решение команды: в индексе и ссылках только .py

# Тексты для модели: описания инструментов и аргументов (правятся здесь, см. docs/mcp_tools.md)
DESCRIPTIONS = {
    "read_file": "Прочитать строки файла репозитория с номерами.",
}
ARGS = {
    "path": "Путь к файлу относительно корня репозитория, например app.py или sansio/scaffold.py.",
    "start_line": "Первая строка (с 1). По умолчанию 1.",
    "end_line": f"Последняя строка включительно. По умолчанию start_line + {DEFAULT_LINES - 1}; "
                f"за один вызов не больше {MAX_LINES} строк.",
}


class ToolInputError(ValueError):
    """Ошибка в аргументах инструмента; текст показывается модели."""


def get_root() -> Path:
    """Корень репозитория: REPO_ROOT из окружения или Config.repo_root."""
    env = os.environ.get("REPO_ROOT")
    if env:
        return Path(env).resolve()
    from ..config import Config
    return Path(Config.from_env().repo_root).resolve()


def resolve_path(root: Path, path: str) -> Path:
    """Путь внутри root. Запрещены пустой, абсолютный путь, "..", "~" и симлинки, ведущие наружу."""
    raw = (path or "").strip().replace("\\", "/")
    if not raw:
        raise ToolInputError("не указан путь")
    rel = Path(raw)
    if rel.is_absolute() or raw.startswith("~") or ".." in rel.parts:
        raise ToolInputError(f"путь вне репозитория: {path} (нужен путь относительно корня, без '..')")
    full = (root / rel).resolve()
    if not full.is_relative_to(root):
        raise ToolInputError(f"путь вне репозитория: {path}")
    return full


def _rel(root: Path, full: Path) -> str:
    return full.relative_to(root).as_posix()


def _error(text: str) -> str:
    return f"Ошибка: {text}"


def read_file(root: Path, path: str, start_line: int | None = None, end_line: int | None = None) -> str:
    """Строки файла в формате "  12| текст"; первая строка ответа — ссылка [путь:начало-конец]."""
    root = Path(root).resolve()
    try:
        full = resolve_path(root, path)
    except ToolInputError as ex:
        return _error(str(ex))
    if not full.exists():
        return _error(f"файл не найден: {path}. Найдите путь через search_code.")
    if full.is_dir():
        return _error(f"{path} — каталог, а не файл")
    if full.suffix not in READ_SUFFIXES:
        return _error(f"читать можно только файлы {', '.join(sorted(READ_SUFFIXES))}: {path}")
    try:
        lines = full.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return _error(f"файл не в UTF-8: {path}")
    total = len(lines)
    rel = _rel(root, full)
    if total == 0:
        return f"[{rel}] пустой файл"

    start = max(1, start_line or 1)
    if start > total:
        return _error(f"в {rel} всего {total} строк, start_line={start} за концом файла")
    end = end_line if end_line is not None else start + DEFAULT_LINES - 1
    if end < start:
        return _error(f"start_line ({start}) больше end_line ({end})")
    notes = []
    end = min(end, total)
    if end - start + 1 > MAX_LINES:
        end = start + MAX_LINES - 1
        notes.append(f"за один вызов не больше {MAX_LINES} строк")

    budget = MAX_CHARS - 300  # запас на заголовок и пометку
    body: list[str] = []
    used = 0
    for n in range(start, end + 1):
        line = f"{n:>4}| {lines[n - 1]}"
        if body and used + len(line) + 1 > budget:
            notes.append(f"обрезано по лимиту {MAX_CHARS} символов")
            break
        body.append(line[:budget])
        used += len(line) + 1
    shown_end = start + len(body) - 1
    header = f"[{rel}:{start}-{shown_end}] из {total} строк"
    footer = ""
    if notes:
        footer = f"\n… {'; '.join(notes)}. Продолжение: read_file(path={rel!r}, start_line={shown_end + 1})"
    elif shown_end < total:
        footer = f"\n… дальше строки {shown_end + 1}-{total}: read_file(path={rel!r}, start_line={shown_end + 1})"
    return f"{header}\n" + "\n".join(body) + footer


def create_server(root: Path | None = None) -> MCPServer:
    root = Path(root or get_root()).resolve()
    server = MCPServer(SERVER_NAME, instructions=f"Инструменты чтения кода репозитория (корень {root.name}).",
                       log_level="WARNING")

    @server.tool(name="read_file", description=DESCRIPTIONS["read_file"])
    def _read_file(path: Annotated[str, Field(description=ARGS["path"])],
                   start_line: Annotated[int | None, Field(description=ARGS["start_line"])] = None,
                   end_line: Annotated[int | None, Field(description=ARGS["end_line"])] = None) -> str:
        return read_file(root, path, start_line, end_line)

    return server


def main() -> None:
    create_server().run("stdio")


if __name__ == "__main__":
    main()
