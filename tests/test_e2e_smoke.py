"""Сквозной путь вопрос -> retrieve -> ответ -> проверка ссылок на заглушках. Без Ollama и клона Flask."""
import json

import pytest

from eval.run_eval import TARGET_ROOT, load_questions, main, path_mapper, score
from hopiumcode import cli, orchestrator
from hopiumcode.config import Config
from hopiumcode.models import Answer
from hopiumcode.orchestrator import (
    ToolsWithFallback,
    build_fake_agent,
    retrieve_with_fallback,
    verify_citations,
)
from hopiumcode.stubs import StubTools, stub_retrieve

CFG = Config(max_steps=2, repo_name="Flask")
QUESTIONS = {q["id"]: q for q in load_questions()}


@pytest.fixture(autouse=True)
def fresh_fallbacks(monkeypatch):
    monkeypatch.setattr(orchestrator, "_FALLBACKS", {})


@pytest.mark.parametrize("qid", ["q01", "q05", "q08", "q12"])
def test_fake_pipeline_answers_with_verified_citations(qid):
    agent = build_fake_agent(CFG)
    retrieved = []
    inner = agent.retrieve
    agent.retrieve = lambda q, k, m: retrieved.extend(inner(q, k, m)) or retrieved
    ans = agent.answer(QUESTIONS[qid]["question"], mode="hybrid")
    assert retrieved, "ретривер не вызывался"
    allowed = [(h.chunk.path, h.chunk.start_line, h.chunk.end_line) for h in retrieved]
    assert ans.citations and not ans.bad_citations
    assert verify_citations(" ".join(ans.citations), allowed).ok


def test_eval_main_fake_writes_result(tmp_path, capsys):
    assert main(["--fake", "--only", "q01,q12", "--out-dir", str(tmp_path)]) == 0
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1 and files[0].with_suffix(".md").exists()
    data = json.loads(files[0].read_text(encoding="utf-8"))
    rows = data["questions"]["hybrid"]
    assert [r["id"] for r in rows] == ["q01", "q12"]
    assert set(data["summary"]["hybrid"]["by_type"]) == {"semantic", "trap"}
    assert all(r["citations_verified"] == 1.0 for r in rows)
    assert "Итог по режимам" in capsys.readouterr().out


def test_eval_main_rejects_unknown_id(tmp_path):
    assert main(["--fake", "--only", "q99", "--out-dir", str(tmp_path)]) == 1


def test_default_agent_wiring():
    pytest.importorskip("ollama")
    agent = orchestrator.build_default_agent(CFG)
    assert callable(agent.retrieve)
    assert isinstance(agent.tools, ToolsWithFallback)    # инструменты подключены с откатом на заглушку


def test_retrieve_fallback_warns_once(capsys):
    calls = []

    def real(q, k, m):
        calls.append(q)
        raise NotImplementedError("участник 3")

    r = retrieve_with_fallback(real, stub_retrieve)
    assert r("a", 2, "vector") == stub_retrieve("a", 2, "vector")
    r("b", 2, "vector")
    assert calls == ["a"]                       # после отката настоящий модуль больше не дёргаем
    assert capsys.readouterr().err.count("⚠") == 1


def test_tools_fallback_on_partial_implementation():
    class Partial:
        started = closed = False

        def start(self):
            self.started = True

        def list_tools_for_ollama(self):
            return []

        def call_tool(self, name, args):
            raise NotImplementedError("участник 4")

        def close(self):
            self.closed = True

    real = Partial()
    t = ToolsWithFallback(real, StubTools())
    assert t.list_tools_for_ollama() == [] and real.started
    assert "(заглушка)" in t.call_tool("read_file", {"path": "app.py"})
    assert orchestrator.fallback_warnings()
    t.close()


def test_tools_server_start_failure_runs_without_tools(capsys):
    class Broken(StubTools):
        def start(self):
            raise RuntimeError("MCP-сервер завершился")

    t = ToolsWithFallback(Broken(), StubTools())
    assert t.list_tools_for_ollama() == []                       # без инструментов, но не падаем
    assert t.call_tool("read_file", {"path": "app.py"}).startswith("Ошибка: инструменты недоступны")
    assert "MCP-сервер не запустился" in capsys.readouterr().err
    ans = build_fake_agent(CFG)
    ans.tools = t
    assert ans.answer("Где создаётся сессия?").citations         # агент отвечает по контексту
    t.close()


def test_index_without_sources_hints_fetch(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CG_REPO_ROOT", str(tmp_path))
    assert cli.main(["index"]) == 1
    assert "make fetch" in capsys.readouterr().err


def test_index_falls_back_to_stub_when_module_not_ready(tmp_path, monkeypatch, capsys):
    from hopiumcode.indexing import build

    def not_ready(*args, **kwargs):
        raise NotImplementedError("участник 2")

    (tmp_path / "a.py").write_text("x = 1\n")
    monkeypatch.setenv("CG_REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(build, "build_all", not_ready)
    assert cli.main(["index"]) == 0
    out = capsys.readouterr()
    assert "indexing.build.build_all" in out.err and "1" in out.out


def test_index_reports_ollama_down(tmp_path, monkeypatch, capsys):
    from hopiumcode.indexing import build

    def down(*args, **kwargs):
        raise ConnectionError("refused")

    (tmp_path / "a.py").write_text("x = 1\n")
    monkeypatch.setenv("CG_REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(build, "build_all", down)
    assert cli.main(["index"]) == 2
    assert "ollama serve" in capsys.readouterr().err


def test_path_mapper_to_target_root():
    to_target = path_mapper(TARGET_ROOT / "src" / "flask")
    assert to_target("sansio/app.py") == "src/flask/sansio/app.py"
    assert to_target("src/flask/app.py") == "src/flask/app.py"
    assert path_mapper(TARGET_ROOT.parent)("flask/src/flask/ctx.py") == "src/flask/ctx.py"


def test_score_trap_and_hit_files():
    trap = QUESTIONS["q12"]
    honest = Answer(text="В коде Flask этого не нашёл.")
    assert score(trap, honest, set(), str)["answered"] is True
    cited = Answer(text="Вот реализация [app.py:1-2]", citations=["[app.py:1-2]"])
    assert score(trap, cited, set(), str)["answered"] is False

    q = QUESTIONS["q05"]  # expected_files: src/flask/app.py
    to_target = path_mapper(TARGET_ROOT / "src" / "flask")
    ans = Answer(text="[app.py:904-920]", citations=["[app.py:904-920]"])
    m = score(q, ans, set(), to_target)
    assert m["hit_files"] == 1.0 and m["citations_verified"] == 1.0 and m["answered"] is True
