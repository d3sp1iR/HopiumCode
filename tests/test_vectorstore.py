"""indexing.vectorstore.VectorIndex на Chroma во временной папке с HashEmbedder (без Ollama)."""
import pytest

from hopiumcode.indexing.embed import HashEmbedder
from hopiumcode.indexing.vectorstore import VectorIndex
from hopiumcode.models import Chunk

CHUNKS = [
    Chunk("config.py::parse_config", "config.py", 1, 3, "function", "parse_config",
          "def parse_config(path):\n    return load_yaml(path)", "python"),
    Chunk("app.py::Flask.run", "app.py", 10, 20, "method", "Flask.run",
          "def run(self, host, port):\n    start_server(host, port)", "python"),
    Chunk("app.py::#1", "app.py", 1, 5, "module", None, "import os\nimport sys", "python"),
    Chunk("sansio/scaffold.py::Scaffold.route", "sansio/scaffold.py", 336, 365, "method", "Scaffold.route",
          "def route(self, rule, **options):\n    self.add_url_rule(rule)", "python"),
]


@pytest.fixture
def index(tmp_path):
    idx = VectorIndex(tmp_path / "chroma", embedder=HashEmbedder())
    idx.add(CHUNKS)
    return idx


def test_search_finds_and_restores_full_chunk(index):
    hits = index.search("parse config path", 2)
    assert hits[0].chunk == CHUNKS[0]                          # все поля, включая text и kind
    assert hits[0].via == "vector"
    assert all(0.0 <= h.score <= 1.0 for h in hits)
    assert hits[0].score >= hits[1].score


def test_module_chunk_name_none_restored(index):
    assert next(c for c in index.get_all() if c.id == "app.py::#1").name is None


def test_k_larger_than_index_and_empty(tmp_path, index):
    assert len(index.search("anything", 50)) == len(CHUNKS)
    assert VectorIndex(tmp_path / "empty", embedder=HashEmbedder()).search("x", 5) == []


def test_persistent(tmp_path, index):
    again = VectorIndex(tmp_path / "chroma", embedder=HashEmbedder())
    assert again.count() == len(CHUNKS)
    assert again.search("route rule url", 1)[0].chunk.id == "sansio/scaffold.py::Scaffold.route"


def test_delete_by_path(index):
    index.delete_by_path("app.py")
    assert {c.path for c in index.get_all()} == {"config.py", "sansio/scaffold.py"}


def test_add_is_upsert(index):
    changed = Chunk(CHUNKS[0].id, "config.py", 1, 4, "function", "parse_config", "def parse_config(): new", "python")
    index.add([changed])
    assert index.count() == len(CHUNKS)
    assert next(c for c in index.get_all() if c.id == changed.id) == changed
