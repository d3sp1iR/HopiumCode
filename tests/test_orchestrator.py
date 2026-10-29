from hopiumcode.config import Config
from hopiumcode.models import Chunk, Hit
from hopiumcode.orchestrator import Agent, ModelReply, ToolCall, build_context
from hopiumcode.stubs import ScriptedChat, StubTools, stub_retrieve

CFG = Config(max_steps=4, repo_name="demo")
REF = "[app/db.py:10-24]"


class Seq:
    """Модель, возвращающая заранее заданные ответы; запоминает, что ей отправляли."""
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def chat(self, messages, tools=None):
        self.calls.append((list(messages), tools))
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


class Tools(StubTools):
    def __init__(self, fail=False):
        self.fail, self.called = fail, []

    def call_tool(self, name, args):
        self.called.append((name, args))
        if self.fail:
            raise RuntimeError("сбой диска")
        return "  1| код [docs/usage.md:20-22]"


def agent(model, tools=None):
    return Agent(model, stub_retrieve, tools, CFG)


def test_simple_answer_with_valid_citation():
    a = agent(Seq(ModelReply(f"Подключение в connect {REF}."))).answer("Где подключение?")
    assert a.citations == [REF] and a.bad_citations == [] and a.steps == 1
    assert "Подтверждено ссылок: 1 из 1" in a.text


def test_tool_flow_read_file_allows_its_range():
    m = Seq(ModelReply("", [ToolCall("read_file", {"path": "app/x.py", "start_line": 5, "end_line": 9})]),
            ModelReply("Нашёл [app/x.py:6-7]."))
    t = Tools()
    a = agent(m, t).answer("Покажи файл app/x.py")
    assert a.used_tools == ["read_file"] and a.steps == 2 and a.bad_citations == []
    roles = [x["role"] for x in m.calls[1][0]]
    assert roles[-2:] == ["assistant", "tool"]          # результат инструмента вернулся модели


def test_citation_from_tool_result_is_allowed():
    m = Seq(ModelReply("", [ToolCall("search_code", {"query": "x"})]), ModelReply("См. [docs/usage.md:21-22]."))
    assert agent(m, Tools()).answer("q").bad_citations == []


def test_bad_citation_triggers_one_repair():
    m = Seq(ModelReply("Выдумка [no/such.py:1-2]."), ModelReply(f"Исправлено {REF}."))
    a = agent(m).answer("q")
    assert len(m.calls) == 2 and a.bad_citations == [] and a.citations == [REF]
    assert "no/such.py" in m.calls[1][0][-1]["content"]  # модели назвали плохие ссылки


def test_bad_citation_persisting_is_flagged_in_text():
    m = Seq(ModelReply("Выдумка [no/such.py:1-2]."))
    a = agent(m).answer("q")
    assert a.bad_citations == ["[no/such.py:1-2]"] and "Непроверенные ссылки" in a.text


def test_tool_error_is_returned_to_model_not_raised():
    m = Seq(ModelReply("", [ToolCall("read_file", {"path": "a.py"})]), ModelReply("Не вышло."))
    a = agent(m, Tools(fail=True)).answer("q")
    assert "Ошибка инструмента" in m.calls[1][0][-1]["content"] and a.text.startswith("Не вышло")


def test_step_limit_forces_final_answer_without_tools():
    loop = ModelReply("", [ToolCall("read_file", {"path": "a.py"})])
    final = ModelReply(f"Итог {REF}.")
    m = Seq(*([loop] * 4 + [final]))
    a = Agent(m, stub_retrieve, Tools(), Config(max_steps=4, repo_name="d")).answer("q")
    assert a.steps == 5 and m.calls[-1][1] is None and a.citations == [REF]


def test_tools_not_implemented_runs_without_tools():
    class NotReady(StubTools):                    # клиент инструментов, который ещё не реализован
        def list_tools_for_ollama(self):
            raise NotImplementedError("участник 4")
    m = Seq(ModelReply(f"Ответ {REF}."))
    a = agent(m, NotReady()).answer("q")
    assert a.citations == [REF] and m.calls[0][1] is None


def test_hint_and_graph_facts_in_prompt():
    m = Seq(ModelReply("ok"))
    agent(m).answer("Что вызывает connect?")
    user = m.calls[0][0][1]["content"]
    assert "find_callers" in user and "graph:calls" in user


def test_context_limit_keeps_first_hit():
    big = Chunk("a::b", "a.py", 1, 2, "function", "b", "x" * 5000, "python")
    ctx, _, used = build_context([Hit(big, 1, "vector"), Hit(big, 0.9, "vector")], max_chars=100)
    assert len(used) == 1 and "a.py" in ctx


def test_scripted_chat_end_to_end():
    a = agent(ScriptedChat(), StubTools()).answer("Где ошибка подключения?")
    assert a.bad_citations == [] and a.citations


def _ollama_chat_with(resp, monkeypatch):
    """OllamaChat без настоящего пакета ollama: подменяем клиента."""
    import sys, types
    from hopiumcode.orchestrator import OllamaChat

    class Client:
        def __init__(self, host=None): ...
        def chat(self, **kw): return resp

    monkeypatch.setitem(sys.modules, "ollama", types.SimpleNamespace(Client=Client))
    return OllamaChat("m")


def test_ollama_adapter_parses_dict_response_and_strips_think(monkeypatch):
    resp = {"message": {"content": "<think>рассуждаю</think>Готово", "tool_calls": [
        {"function": {"name": "read_file", "arguments": {"path": "a.py"}}}]}}
    r = _ollama_chat_with(resp, monkeypatch).chat([{"role": "user", "content": "x"}])
    assert r.content == "Готово" and r.tool_calls[0].name == "read_file" and r.tool_calls[0].args == {"path": "a.py"}


def test_ollama_adapter_parses_object_response(monkeypatch):
    from types import SimpleNamespace as NS
    resp = NS(message=NS(content="Привет", tool_calls=[NS(function=NS(name="git_log", arguments='{"path": "a.py"}'))]))
    r = _ollama_chat_with(resp, monkeypatch).chat([])
    assert r.tool_calls[0].args == {"path": "a.py"} and r.content == "Привет"
