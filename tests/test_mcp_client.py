"""McpClient: настоящий MCP-сервер подпроцессом на временном репозитории. Без Ollama и клона Flask."""
import os
import sys
import textwrap
import time
from pathlib import Path

import pytest

from hopiumcode import orchestrator
from hopiumcode.mcp_client import McpClient, to_ollama_tool

TOOLS = {"read_file", "search_code", "find_callers", "git_log", "git_blame"}


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "app.py").write_text("def a():\n    return b()\n\n\ndef b():\n    return 1\n", encoding="utf-8")
    return root


def _children() -> set[int]:
    me, out = str(os.getpid()), set()
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat.read_text().rsplit(")", 1)[1].split()
        except OSError:
            continue
        if fields[1] == me and fields[0] != "Z":
            out.add(int(stat.parent.name))
    return out


def test_to_ollama_tool_simplifies_schema():
    schema = {"type": "object", "title": "x", "required": ["path"], "properties": {
        "path": {"type": "string", "title": "Path", "description": "путь"},
        "start_line": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": None, "description": "с"}}}
    tool = to_ollama_tool("read_file", "читать", schema)
    assert tool == {"type": "function", "function": {"name": "read_file", "description": "читать", "parameters": {
        "type": "object", "required": ["path"], "properties": {
            "path": {"type": "string", "description": "путь"},
            "start_line": {"type": "integer", "default": None, "description": "с"}}}}}


def test_round_trip(repo):
    with McpClient(root=repo) as client:
        tools = client.list_tools_for_ollama()
        assert {t["function"]["name"] for t in tools} == TOOLS
        read = next(t for t in tools if t["function"]["name"] == "read_file")["function"]
        assert read["parameters"]["required"] == ["path"]
        assert read["parameters"]["properties"]["start_line"]["type"] == "integer"
        assert client.call_tool("read_file", {"path": "app.py", "start_line": 5}).startswith("[app.py:5-6]")
        assert client.call_tool("read_file", {"path": "../etc/passwd"}).startswith("Ошибка: путь вне репозитория")
        assert "[app.py:1-2] | a" in client.call_tool("find_callers", {"symbol": "b"})
        assert client.call_tool("no_such_tool", {}).startswith("Ошибка")
        assert client.call_tool("read_file", {}).startswith("Ошибка")      # нет обязательного аргумента


def test_lazy_start_and_restart_after_close(repo):
    client = McpClient(root=repo)
    assert client._session is None                              # ничего не запущено до первого обращения
    assert "def a" in client.call_tool("read_file", {"path": "app.py"})
    client.close()
    client.close()                                              # повторное закрытие безопасно
    assert "def a" in client.call_tool("read_file", {"path": "app.py"})
    client.close()


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="нужен /proc (Linux)")
def test_close_terminates_server_process(repo):
    before = _children()
    client = McpClient(root=repo)
    client.start()
    started = _children() - before
    assert started, "сервер не запущен подпроцессом"
    client.close()
    deadline = time.time() + 5
    while _children() & started and time.time() < deadline:
        time.sleep(0.1)
    assert not (_children() & started), "подпроцесс сервера остался после close()"


def test_server_that_fails_to_start(capsys):
    client = McpClient(command=[sys.executable, "-c", "import sys; sys.exit(3)"], timeout=10)
    client.start()                                              # без исключения
    assert client.list_tools_for_ollama() == []
    assert client.call_tool("read_file", {"path": "a.py"}).startswith("Ошибка: MCP-сервер")
    assert "⚠ MCP-сервер" in capsys.readouterr().err
    client.close()


def test_tool_timeout_returns_text(tmp_path):
    script = tmp_path / "slow_server.py"
    script.write_text(textwrap.dedent('''\
        import time
        from mcp.server.mcpserver import MCPServer
        s = MCPServer("slow", log_level="WARNING")

        @s.tool()
        def slow() -> str:
            time.sleep(5)
            return "поздно"

        s.run("stdio")
    '''), encoding="utf-8")
    with McpClient(command=[sys.executable, str(script)], timeout=1) as client:
        t = time.perf_counter()
        assert client.call_tool("slow", {}) == "Ошибка: инструмент slow не ответил за 1 c"
        assert time.perf_counter() - t < 4


def test_default_agent_uses_mcp_without_fallback(repo, monkeypatch, capsys):
    pytest.importorskip("ollama")
    monkeypatch.setenv("REPO_ROOT", str(repo))
    monkeypatch.setattr(orchestrator, "_FALLBACKS", {})
    agent = orchestrator.build_default_agent()
    try:
        assert {t["function"]["name"] for t in agent.tools.list_tools_for_ollama()} == TOOLS
        assert "def b" in agent.tools.call_tool("read_file", {"path": "app.py"})
    finally:
        agent.tools.close()
    assert not any("McpClient" in w for w in orchestrator.fallback_warnings())
    assert "McpClient" not in capsys.readouterr().err
