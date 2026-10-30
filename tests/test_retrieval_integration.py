"""Настоящие chunker + BM25 + граф + Retriever на мини-репозитории; векторная часть подставная."""
from hopiumcode.graph.extract import extract
from hopiumcode.graph.store import NetworkXStore
from hopiumcode.indexing.chunker import chunk_file
from hopiumcode.indexing.loader import iter_files
from hopiumcode.indexing.vectorstore import VectorIndex
from hopiumcode.config import Config
from hopiumcode.retrieval.hybrid import Retriever
from hopiumcode.retrieval.lexical import Lexical

from .fakes import FakeCollection, FakeEmbedder

FILES = {
    "net.py": "def open_socket(addr):\n    '''Открывает сокет и возвращает соединение.'''\n    return addr\n",
    "client.py": "from net import open_socket\n\n\ndef fetch(url):\n    conn = open_socket(url)\n    return conn\n\n\ndef ping(url):\n    return open_socket(url)\n",
    "render.py": "def render(template):\n    return template.upper()\n",
    "util.py": "def noop():\n    return None\n",
}


def build(tmp_path):
    for rel, text in FILES.items():
        (tmp_path / rel).write_text(text)
    chunks = [c for p in iter_files(tmp_path) for c in chunk_file(p, tmp_path)]
    vector = VectorIndex(Config(index_dir=tmp_path / "idx"), FakeEmbedder(), FakeCollection())
    vector.add(chunks)
    store = NetworkXStore(tmp_path / "graph.json")
    nodes, edges = extract(tmp_path)
    store.add_nodes(nodes)
    store.add_edges(edges)
    return Retriever(vector, Lexical(chunks), store, {c.id: c for c in chunks}), chunks


def test_modes_and_graph_adds_callers(tmp_path):
    retriever, _ = build(tmp_path)
    question = "open_socket addr"
    vec = retriever.retrieve(question, k=1, mode="vector")
    assert [h.chunk.id for h in vec] == ["net.py::open_socket"]          # в узком окне вызывающих нет

    both = retriever.retrieve(question, k=4, mode="vector+bm25")
    assert all(h.via in ("vector", "bm25", "vector+bm25") for h in both)

    hybrid = retriever.retrieve(question, k=6, mode="hybrid")
    by_id = {h.chunk.id: h for h in hybrid}
    assert "client.py::fetch" in by_id and "client.py::ping" in by_id    # вызывающие пришли из графа
    assert by_id["client.py::fetch"].via.startswith("graph:calls") or "graph" in by_id["client.py::fetch"].via
    assert all(0 <= h.score <= 1 for h in hybrid)


def test_callers_survive_char_budget(tmp_path):
    # Узкий лимит символов: без квоты графа вызывающие вытеснялись прямыми совпадениями поиска.
    retriever, _ = build(tmp_path)
    retriever.max_chars = 120
    hybrid = retriever.retrieve("open_socket addr", k=4, mode="hybrid")
    ids = [h.chunk.id for h in hybrid]
    assert ids[0] == "net.py::open_socket"
    assert {"client.py::fetch", "client.py::ping"} <= set(ids)


def test_query_expansion_is_applied(tmp_path):
    retriever, _ = build(tmp_path)
    seen = []
    retriever.expand_query = lambda q: seen.append(q) or q + " open_socket"
    hits = retriever.retrieve("где открывается сокет", k=2, mode="vector+bm25")
    assert seen == ["где открывается сокет"] and hits[0].chunk.id == "net.py::open_socket"


def test_unknown_mode(tmp_path):
    retriever, _ = build(tmp_path)
    try:
        retriever.retrieve("x", mode="magic")
    except ValueError:
        pass
    else:
        raise AssertionError("ожидали ValueError")
