"""Тесты search_code и find_callers на заглушках retrieve() и GraphStore.

Настоящие модули участников 2 и 3 не нужны: подменяются функции
server._retrieve и server._get_graph, данные — по контрактам Chunk/Hit (ТЗ, раздел 5).
"""

from types import SimpleNamespace

import pytest

from hopiumcode.mcp_server import server
from hopiumcode.mcp_server.server import (
    MAX_CHARS,
    ToolInputError,
    find_callers_impl,
    search_code_impl,
)

SESSIONS = '''\
class SessionInterface:
    def open_session(self, app, request):
        raise NotImplementedError


class SecureCookieSessionInterface(SessionInterface):
    @staticmethod
    def helper():
        return 1

    def open_session(self, app, request):
        s = self.get_signing_serializer(app)
        return s
'''

APP = '''\
def preprocess():
    pass


class Flask:
    def make_response(self, rv):
        return rv

    def wsgi_app(self, environ):
        ctx = self.request_context(environ)
        return ctx.push()
'''


def hit(path, start, end, name, text, via="vector", score=0.9, kind="function"):
    chunk = SimpleNamespace(id=f"{path}::{name}", path=path, start_line=start, end_line=end,
                            kind=kind, name=name, text=text, lang="python")
    return SimpleNamespace(chunk=chunk, score=score, via=via)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    (tmp_path / "sessions.py").write_text(SESSIONS, encoding="utf-8")
    (tmp_path / "app.py").write_text(APP, encoding="utf-8")
    monkeypatch.setenv("REPO_ROOT", str(tmp_path))
    monkeypatch.delenv("CG_REPO_ROOT", raising=False)
    original = server._get_graph  # тесты подменяют его через monkeypatch
    original.cache_clear()
    yield tmp_path
    original.cache_clear()


# --- search_code --------------------------------------------------------------------
def test_search_code_format(repo, monkeypatch):
    calls = []

    def fake_retrieve(query, k):
        calls.append((query, k))
        return [
            hit("sessions.py", 11, 13, "SecureCookieSessionInterface.open_session",
                "    def open_session(self, app, request):\n"
                "        s = self.get_signing_serializer(app)\n        return s",
                via="vector+bm25", score=0.83),
            hit("app.py", 9, 11, "Flask.wsgi_app", "def wsgi_app(...):\n    ...",
                via="graph:calls"),
        ]

    monkeypatch.setattr(server, "_retrieve", fake_retrieve)
    out = search_code_impl("подпись cookie сессии", k=3)
    assert calls == [("подпись cookie сессии", 3)]
    first, second = out.split("\n\n")
    assert first.splitlines() == [
        "[sessions.py:11-13] SecureCookieSessionInterface.open_session (vector+bm25, 0.83)",
        " 11|    def open_session(self, app, request):",
        " 12|        s = self.get_signing_serializer(app)",
        " 13|        return s",
    ]
    assert second.startswith("[app.py:9-11] Flask.wsgi_app (graph:calls, 0.90)")


def test_search_code_long_chunk_shows_start_and_read_file_hint(repo, monkeypatch):
    text = "\n".join(f"line {i}" for i in range(40))
    monkeypatch.setattr(server, "_retrieve", lambda q, k: [hit("app.py", 100, 139, "big", text)])
    lines = search_code_impl("x").splitlines()
    assert len(lines) == 1 + server.SNIPPET_LINES + 1
    assert 'read_file(path="app.py", start_line=100, end_line=139)' in lines[-1]


def test_search_code_k_bounds(repo, monkeypatch):
    seen = []
    monkeypatch.setattr(server, "_retrieve", lambda q, k: seen.append(k) or [])
    assert search_code_impl("x", k=100).startswith("ничего не найдено")
    assert seen == [server.SEARCH_K_MAX]
    with pytest.raises(ToolInputError, match="k должен быть"):
        search_code_impl("x", k=0)
    with pytest.raises(ToolInputError, match="пустой запрос"):
        search_code_impl("   ")


def test_search_code_output_capped(repo, monkeypatch):
    hits = [hit("app.py", i, i + 5, f"f{i}", ("x" * 300 + "\n") * 6) for i in range(10)]
    monkeypatch.setattr(server, "_retrieve", lambda q, k: hits)
    out = search_code_impl("x", k=10)
    assert len(out) <= MAX_CHARS and "обрезано" in out


@pytest.mark.parametrize("exc", [ImportError("no module"), NotImplementedError()])
def test_search_code_not_ready(repo, monkeypatch, exc):
    def broken(q, k):
        raise exc

    monkeypatch.setattr(server, "_retrieve", broken)
    with pytest.raises(ToolInputError, match="поиск ещё не готов"):
        search_code_impl("x")


# --- find_callers -------------------------------------------------------------------
class FakeGraph:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def callers(self, symbol, depth=1):
        self.calls.append((symbol, depth))
        return self.result


def use_graph(monkeypatch, graph):
    monkeypatch.setattr(server, "_get_graph", lambda: graph)


def test_find_callers_with_lines_from_ast(repo, monkeypatch):
    g = FakeGraph(["app.py::Flask.wsgi_app", "sessions.py::SecureCookieSessionInterface.helper",
                   "app.py::preprocess", "app.py::Flask.wsgi_app"])  # дубликат схлопывается
    use_graph(monkeypatch, g)
    out = find_callers_impl("open_session()")
    assert g.calls == [("open_session", 1)]
    assert out.splitlines() == [
        "[app.py:9-11] Flask.wsgi_app",
        "[sessions.py:7-9] SecureCookieSessionInterface.helper",  # с декоратором
        "[app.py:1-2] preprocess",
    ]


def test_find_callers_unknown_symbol_location(repo, monkeypatch):
    use_graph(monkeypatch, FakeGraph(["gone.py::old_func", "app.py::Missing.method", "raw-id"]))
    assert find_callers_impl("x").splitlines() == [
        "[gone.py] old_func", "[app.py] Missing.method", "raw-id"
    ]


def test_find_callers_ids_cannot_escape_repo(repo, monkeypatch, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside")
    (outside / "evil.py").write_text("def f():\n    pass\n", encoding="utf-8")
    use_graph(monkeypatch, FakeGraph([f"../{outside.name}/evil.py::f", f"{outside}/evil.py::f"]))
    for line in find_callers_impl("x").splitlines():
        assert ":1-2]" not in line  # файл снаружи не читался


def test_find_callers_empty_and_depth(repo, monkeypatch):
    g = FakeGraph([])
    use_graph(monkeypatch, g)
    assert find_callers_impl("nothing", depth=10) == "вызовов не найдено: nothing"
    assert g.calls == [("nothing", server.CALLERS_DEPTH_MAX)]
    with pytest.raises(ToolInputError, match="depth должен быть"):
        find_callers_impl("x", depth=0)


def test_find_callers_graph_not_ready(repo, monkeypatch):
    def missing():
        raise ImportError("no graph.store")

    monkeypatch.setattr(server, "_get_graph", missing)
    with pytest.raises(ToolInputError, match="граф кода ещё не готов"):
        find_callers_impl("open_session")


def test_find_callers_graph_not_built(repo, monkeypatch):
    def not_built():
        raise FileNotFoundError("граф не построен: data/graph/graph.json")

    monkeypatch.setattr(server, "_get_graph", not_built)
    with pytest.raises(ToolInputError, match="make index"):
        find_callers_impl("open_session")


def test_search_code_index_not_built(repo, monkeypatch):
    from hopiumcode.indexing.vectorstore import IndexNotBuilt

    def empty(query, k):
        raise IndexNotBuilt("Векторный индекс пуст: выполните `make index`")

    monkeypatch.setattr(server, "_retrieve", empty)
    with pytest.raises(ToolInputError, match="make index"):
        server.search_code_impl("сессия")


def test_get_graph_uses_participant3_loader(tmp_path, monkeypatch):
    from hopiumcode.graph import query
    sentinel = object()
    monkeypatch.setattr(query, "get_store", lambda: sentinel)
    assert server._get_graph() is sentinel
