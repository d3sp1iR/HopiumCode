"""Заглушки, чтобы отлаживать агента до готовности чужих модулей. Владелец: участник 1.

StubRetrieve, StubTools и ScriptedChat повторяют контракты настоящих модулей. Образец формата для участников 2-4.
"""
from __future__ import annotations

from pathlib import Path

from .models import Chunk, Hit
from .orchestrator import ModelReply

_DEMO = [
    Chunk("app/db.py::Connection.connect", "app/db.py", 10, 24, "method", "Connection.connect",
          "def connect(self):\n    try:\n        self._sock = socket.create_connection(self.addr)\n"
          "    except OSError as ex:\n        raise ConnectionFailed(ex)", "python"),
    Chunk("app/errors.py::ConnectionFailed", "app/errors.py", 3, 8, "class", "ConnectionFailed",
          "class ConnectionFailed(Exception):\n    \"\"\"Ошибка подключения к серверу.\"\"\"", "python"),
    Chunk("docs/usage.md::#2", "docs/usage.md", 20, 30, "doc", None,
          "## Ошибки\nПри недоступности сервера выбрасывается ConnectionFailed.", "markdown"),
]


def stub_retrieve(question: str, k: int = 8, mode: str = "hybrid") -> list[Hit]:
    hits = [Hit(c, 0.9 - 0.1 * i, "vector") for i, c in enumerate(_DEMO)]
    if mode == "hybrid":
        hits[-1] = Hit(_DEMO[-1], 0.5, "graph:calls")
    return hits[:k]


class StubTools:
    """Тот же интерфейс, что у McpClient (участник 4)."""

    def start(self) -> None: ...

    def list_tools_for_ollama(self) -> list[dict]:
        return [{"type": "function", "function": {
            "name": "read_file", "description": "Прочитать файл репозитория (только чтение)",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}},
                "required": ["path"]}}}]

    def call_tool(self, name: str, args: dict) -> str:
        if name == "read_file":
            return f"  1| (заглушка) содержимое {args.get('path')}"
        return f"(заглушка) инструмент {name} не реализован"

    def close(self) -> None: ...


class ScriptedChat:
    """Модель-заглушка: отвечает по контексту из последнего пользовательского сообщения, без LLM."""

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> ModelReply:
        user = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        first_ref = user.split("[", 1)[1].split("]", 1)[0] if "[" in user else None
        if first_ref:
            return ModelReply(f"(заглушка) Ответ по найденному фрагменту [{first_ref}].")
        return ModelReply("В найденном контексте ответа нет.")


def stub_build_all(root: Path) -> int:
    """Заглушка indexing.build.build_all: индекс не строит, только считает .py-файлы, которые попадут в индекс."""
    n = sum(1 for p in root.rglob("*.py") if "__pycache__" not in p.parts)
    print(f"(заглушка) индекс не построен; в {root} найдено .py-файлов для индексации: {n}")
    return n
