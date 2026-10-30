"""Тесты McpClient.

Основная часть работает с поддельным MCP-сервером (простой скрипт на JSON-RPC):
так проверяются таймаут, падение сервера, пагинация и ping без Ollama и без SDK.
В конце — интеграционный тест с настоящим hopiumcode.mcp_server.server.
"""

import functools
import json
import sys
import textwrap
import time
from pathlib import Path

import pytest

from hopiumcode import mcp_client
from hopiumcode.mcp_client import McpClient, McpClientError, _result_to_text, mcp_tool_to_ollama

FAKE_SERVER = textwrap.dedent(r'''
    import json, os, sys, time

    OPT_INT = {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": None,
               "title": "Start Line", "description": "Первая строка."}
    TOOLS = [
        {"name": "echo", "description": "Повторяет текст.",
         "inputSchema": {"type": "object", "title": "echoArguments",
                         "properties": {"text": {"type": "string", "title": "Text"},
                                        "start_line": OPT_INT},
                         "required": ["text"]}},
        {"name": "add", "description": "Сумма.",
         "inputSchema": {"type": "object",
                         "properties": {"a": {"type": "integer"},
                                        "b": {"type": "integer", "default": 1}},
                         "required": ["a"]}},
        {"name": "fail", "description": "", "inputSchema": {"type": "object", "properties": {}}},
        {"name": "slow", "description": "", "inputSchema": {"type": "object", "properties": {}}},
        {"name": "crash", "description": "", "inputSchema": {"type": "object", "properties": {}}},
        {"name": "env", "description": "", "inputSchema": {"type": "object", "properties": {}}},
        {"name": "chatty", "description": "", "inputSchema": {"type": "object", "properties": {}}},
    ]

    def send(msg):
        sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    def text(req_id, value, is_error=False):
        send({"jsonrpc": "2.0", "id": req_id,
              "result": {"content": [{"type": "text", "text": value}], "isError": is_error}})

    print("fake server started", file=sys.stderr, flush=True)
    for line in sys.stdin:
        msg = json.loads(line)
        method, req_id, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
        if req_id is None:
            continue  # уведомления клиента
        if method == "initialize":
            send({"jsonrpc": "2.0", "id": req_id, "result": {
                "protocolVersion": params["protocolVersion"], "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake", "version": "1"}}})
        elif method == "tools/list":  # две страницы
            page = TOOLS[:3] if "cursor" not in params else TOOLS[3:]
            result = {"tools": page}
            if "cursor" not in params:
                result["nextCursor"] = "p2"
            send({"jsonrpc": "2.0", "id": req_id, "result": result})
        elif method == "tools/call":
            name, args = params["name"], params.get("arguments", {})
            if name == "echo":
                text(req_id, json.dumps(args, ensure_ascii=False, sort_keys=True))
            elif name == "add":
                if not all(isinstance(v, int) for v in args.values()):
                    text(req_id, "Error executing tool add: not int", True)
                else:
                    text(req_id, str(args["a"] + args.get("b", 1)))
            elif name == "fail":
                text(req_id, "Error executing tool fail: путь вне репозитория", True)
            elif name == "slow":
                time.sleep(30)
            elif name == "crash":
                print("boom: сервер упал", file=sys.stderr, flush=True)
                sys.exit(1)
            elif name == "env":
                text(req_id, os.environ.get("REPO_ROOT", "-"))
            elif name == "chatty":
                print("отладочный print в stdout")  # не JSON — клиент должен пропустить
                send({"jsonrpc": "2.0", "method": "notifications/message",
                      "params": {"level": "info", "data": "x"}})
                send({"jsonrpc": "2.0", "id": "srv-1", "method": "ping"})
                pong = json.loads(sys.stdin.readline())
                text(req_id, "pong ok" if pong.get("result") == {} else "no pong")
            else:
                send({"jsonrpc": "2.0", "id": req_id,
                      "error": {"code": -32602, "message": f"Unknown tool: {name}"}})
        else:
            send({"jsonrpc": "2.0", "id": req_id,
                  "error": {"code": -32601, "message": "Method not found"}})
''')


@pytest.fixture
def fake_cmd(tmp_path):
    script = tmp_path / "fake_server.py"
    script.write_text(FAKE_SERVER, encoding="utf-8")
    return [sys.executable, str(script)]


@pytest.fixture
def client(fake_cmd):
    c = McpClient(command=fake_cmd, env={"REPO_ROOT": "/repo"}, timeout=2)
    c.start()
    yield c
    c.close()


# --- запуск и список инструментов ----------------------------------------------
def test_start_handshake(client):
    assert client.is_running
    assert client.server_info["name"] == "fake"


def test_list_tools_paginated_and_converted(client):
    tools = client.list_tools_for_ollama()
    assert [t["function"]["name"] for t in tools] == [
        "echo", "add", "fail", "slow", "crash", "env", "chatty"
    ]
    echo = tools[0]
    assert echo == {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Повторяет текст.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "start_line": {"type": "integer", "description": "Первая строка."},
                },
                "required": ["text"],
            },
        },
    }
    assert tools[1]["function"]["parameters"]["properties"]["b"] == {
        "type": "integer", "description": "По умолчанию 1."
    }


def test_convert_ref_and_type_list():
    tool = {
        "name": "t",
        "inputSchema": {
            "type": "object",
            "properties": {
                "mode": {"$ref": "#/$defs/Mode", "description": "Режим."},
                "n": {"type": ["integer", "null"]},
            },
            "required": ["mode"],
            "$defs": {"Mode": {"type": "string", "enum": ["vector", "bm25"], "title": "Mode"}},
        },
    }
    params = mcp_tool_to_ollama(tool)["function"]["parameters"]
    assert params["properties"]["mode"] == {
        "type": "string", "description": "Режим.", "enum": ["vector", "bm25"]
    }
    assert params["properties"]["n"] == {"type": "integer"}
    assert params["required"] == ["mode"]


# --- вызовы ---------------------------------------------------------------------
def test_result_keeps_leading_spaces():
    # формат read_file " 2|код": ведущий пробел нельзя срезать
    res = {"content": [{"type": "text", "text": " 2|b = 2\n 3|c = 3\n"}]}
    assert _result_to_text("read_file", res) == " 2|b = 2\n 3|c = 3"
    assert _result_to_text("x", {"content": [{"type": "text", "text": "  \n"}]}) == (
        "(пустой результат)"
    )



def test_call_tool_text(client):
    client.list_tools_for_ollama()
    assert json.loads(client.call_tool("echo", {"text": "привет"})) == {"text": "привет"}


def test_args_coercion_for_small_models(client):
    client.list_tools_for_ollama()  # схемы нужны для приведения типов
    assert client.call_tool("add", {"a": "2", "b": "3"}) == "5"
    assert client.call_tool("add", '{"a": 2.0}') == "3"            # строка JSON, float -> int
    out = json.loads(client.call_tool("echo", {"text": 7, "start_line": None, "junk": 1}))
    assert out == {"text": "7"}                                     # None и лишнее выброшены


def test_bad_json_args(client):
    assert client.call_tool("echo", "{text: ").startswith("Ошибка: аргументы не являются JSON")


def test_tool_error_is_text(client):
    assert client.call_tool("fail") == "Ошибка: путь вне репозитория"


def test_unknown_tool(client):
    assert client.call_tool("nope", {}) == "Ошибка: Unknown tool: nope"


def test_server_ping_and_noise(client):
    assert client.call_tool("chatty") == "pong ok"


def test_env_passed(client):
    assert client.call_tool("env") == "/repo"


def test_timeout_returns_text_and_restarts(client):
    client.list_tools_for_ollama()
    t0 = time.monotonic()
    out = client.call_tool("slow")
    assert out == "Ошибка: инструмент slow не ответил за 2 с"
    assert time.monotonic() - t0 < 10
    assert client.is_running  # сервер перезапущен
    assert client.call_tool("add", {"a": "1"}) == "2"  # и кэш схем восстановлен


def test_server_crash(client):
    out = client.call_tool("crash")
    assert out.startswith("Ошибка: MCP-сервер завершился")
    assert "boom" in out
    assert client.call_tool("echo", {"text": "x"}).startswith("Ошибка: MCP-сервер не запущен")


def test_default_timeout_is_20_seconds():
    assert McpClient().timeout == 20


# --- запуск и остановка -----------------------------------------------------------
def test_start_fails_with_server_stderr(tmp_path):
    script = tmp_path / "bad.py"
    script.write_text(
        "import sys\nprint('не задана переменная окружения REPO_ROOT', file=sys.stderr)\n"
        "sys.exit(1)\n",
        encoding="utf-8",
    )
    c = McpClient(command=[sys.executable, str(script)], timeout=5)
    with pytest.raises(McpClientError, match="REPO_ROOT"):
        c.start()
    assert not c.is_running


def test_start_command_not_found():
    with pytest.raises(McpClientError, match="не удалось запустить"):
        McpClient(command=["/nonexistent/python-mcp"]).start()


def test_close_idempotent_and_context_manager(fake_cmd):
    with McpClient(command=fake_cmd) as c:
        assert c.is_running
    assert not c.is_running
    c.close()
    assert c.call_tool("echo", {"text": "x"}).startswith("Ошибка: MCP-сервер не запущен")


def test_lazy_start_without_explicit_start(fake_cmd):
    # так клиента использует оркестратор: McpClient() и сразу методы
    c = McpClient(command=fake_cmd)
    try:
        assert len(c.list_tools_for_ollama()) == 7
        assert c.call_tool("add", {"a": "1"}) == "2"
    finally:
        c.close()
    c2 = McpClient(command=fake_cmd)
    try:
        assert not c2.call_tool("env").startswith("Ошибка")  # call_tool тоже поднимает сервер
        assert c2.is_running
    finally:
        c2.close()


def test_lazy_start_failure(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("import sys\nprint('нет REPO_ROOT', file=sys.stderr)\nsys.exit(1)\n",
                   encoding="utf-8")
    c = McpClient(command=[sys.executable, str(bad)], timeout=5)
    with pytest.raises(McpClientError, match="нет REPO_ROOT"):
        c.list_tools_for_ollama()           # список инструментов: ошибка исключением
    out = c.call_tool("read_file", {"path": "app.py"})
    assert out.startswith("Ошибка") and "нет REPO_ROOT" in out  # вызов: ошибка текстом


# --- модульные функции для оркестратора ---------------------------------------------
def test_module_level_api_restarts_after_crash(fake_cmd, monkeypatch):
    monkeypatch.setattr(mcp_client, "McpClient", functools.partial(McpClient, command=fake_cmd))
    monkeypatch.setattr(mcp_client, "_default_client", None)
    try:
        assert len(mcp_client.list_tools_for_ollama()) == 7
        assert mcp_client.call_tool("add", {"a": "4"}) == "5"
        assert mcp_client.call_tool("crash").startswith("Ошибка: MCP-сервер завершился")
        assert mcp_client.call_tool("add", {"a": "1"}) == "2"  # общий клиент поднялся заново
    finally:
        mcp_client.close()


def test_module_level_call_tool_never_raises(tmp_path, monkeypatch):
    bad = tmp_path / "bad.py"
    bad.write_text("import sys\nprint('нет REPO_ROOT', file=sys.stderr)\nsys.exit(1)\n",
                   encoding="utf-8")
    monkeypatch.setattr(mcp_client, "McpClient",
                        functools.partial(McpClient, command=[sys.executable, str(bad)]))
    monkeypatch.setattr(mcp_client, "_default_client", None)
    out = mcp_client.call_tool("read_file", {"path": "app.py"})
    assert out.startswith("Ошибка") and "нет REPO_ROOT" in out


# --- интеграция с настоящим сервером ------------------------------------------------
def test_real_server(tmp_path):
    pytest.importorskip("mcp")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("a = 1\nb = 2\nc = 3\n", encoding="utf-8")
    src = Path(__file__).resolve().parents[1] / "src"
    env = {"REPO_ROOT": str(repo), "PYTHONPATH": str(src)}

    with McpClient(env=env) as c:
        tools = {t["function"]["name"]: t["function"] for t in c.list_tools_for_ollama()}
        assert set(tools) == {"search_code", "read_file", "find_callers", "git_log", "git_blame"}
        rf = tools["read_file"]
        assert "НЕ используй" in rf["description"]
        assert rf["parameters"]["required"] == ["path"]
        assert rf["parameters"]["properties"]["start_line"]["type"] == "integer"

        assert c.call_tool("read_file", {"path": "app.py", "start_line": "2"}) == (
            " 2|b = 2\n 3|c = 3"
        )
        err = c.call_tool("read_file", {"path": "../../etc/passwd"})
        assert err == "Ошибка: путь вне репозитория: «..» в пути запрещено"
