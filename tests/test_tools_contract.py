"""Контракт клиента инструментов (interfaces.ToolClientLike) для McpClient и StubTools.

Пока McpClient бросает NotImplementedError (участник 4), его проверки пропускаются;
StubTools проверяется всегда как образец формата.
"""
import pytest

from hopiumcode.interfaces import ToolClientLike
from hopiumcode.mcp_client import McpClient
from hopiumcode.stubs import StubTools

# инструменты, перечисленные в docstring mcp_server/server.py
DOCUMENTED_TOOLS = {"read_file", "search_code", "find_callers", "git_log", "git_blame"}
METHODS = ("start", "list_tools_for_ollama", "call_tool", "close")


@pytest.fixture
def real_client():
    client = McpClient()
    try:
        client.start()
    except NotImplementedError as ex:
        pytest.skip(f"McpClient ещё не реализован ({ex}): контракт проверится после слияния зоны 4")
    yield client
    client.close()


def _check_schemas(schemas: list[dict]) -> set[str]:
    assert isinstance(schemas, list) and schemas
    names = set()
    for s in schemas:
        assert s["type"] == "function"
        fn = s["function"]
        assert isinstance(fn["name"], str) and fn["name"]
        assert isinstance(fn.get("description", ""), str)
        params = fn["parameters"]
        assert params["type"] == "object" and isinstance(params.get("properties", {}), dict)
        assert set(params.get("required", [])) <= set(params.get("properties", {}))
        names.add(fn["name"])
    assert len(names) == len(schemas), "имена инструментов повторяются"
    return names


@pytest.mark.parametrize("cls", [McpClient, StubTools])
def test_has_tool_client_methods(cls):
    for m in METHODS:
        assert callable(getattr(cls, m, None)), f"{cls.__name__}.{m}"


def test_stub_tools_follow_contract():
    t: ToolClientLike = StubTools()
    t.start()
    names = _check_schemas(t.list_tools_for_ollama())
    assert "read_file" in names
    assert isinstance(t.call_tool("read_file", {"path": "app.py"}), str)
    assert isinstance(t.call_tool("нет_такого", {}), str)   # неизвестный инструмент не роняет агента
    t.close()


def test_real_client_lists_documented_tools(real_client):
    names = _check_schemas(real_client.list_tools_for_ollama())
    missing = DOCUMENTED_TOOLS - names
    assert not missing, f"нет инструментов из mcp_server/server.py: {sorted(missing)}"


def test_real_client_returns_text(real_client):
    out = real_client.call_tool("read_file", {"path": "app.py", "start_line": 1, "end_line": 5})
    assert isinstance(out, str) and len(out) <= 8000           # лимит ответа из mcp_server/server.py


def test_real_client_refuses_paths_outside_repo(real_client):
    try:
        out = real_client.call_tool("read_file", {"path": "../../../../etc/passwd"})
    except Exception:  # noqa: BLE001 - отказ исключением тоже допустим, агент вернёт его модели текстом
        return
    assert "root:" not in out
