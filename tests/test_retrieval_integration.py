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


def test_unknown_mode(tmp_path):
    retriever, _ = build(tmp_path)
    try:
        retriever.retrieve("x", mode="magic")
    except ValueError:
        pass
    else:
        raise AssertionError("ожидали ValueError")
