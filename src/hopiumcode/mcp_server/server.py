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

from mcp.server.mcpserver import MCPServer

SERVER_NAME = "hopiumcode"
MAX_CHARS = 8000


def get_root() -> Path:
    """Корень репозитория: REPO_ROOT из окружения или Config.repo_root."""
    env = os.environ.get("REPO_ROOT")
    if env:
        return Path(env).resolve()
    from ..config import Config
    return Path(Config.from_env().repo_root).resolve()


def create_server(root: Path | None = None) -> MCPServer:
    root = Path(root or get_root()).resolve()
    server = MCPServer(SERVER_NAME, instructions=f"Инструменты чтения кода репозитория (корень {root.name}).",
                       log_level="WARNING")
    return server


def main() -> None:
    create_server().run("stdio")


if __name__ == "__main__":
    main()
