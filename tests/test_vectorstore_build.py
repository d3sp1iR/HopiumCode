"""Векторный индекс на подставных эмбеддингах и инкрементальная сборка (участник 2)."""
import json

from hopiumcode.config import Config
from hopiumcode.indexing.build import build_all
from hopiumcode.indexing.chunker import chunk_file
from hopiumcode.indexing.embed import OllamaEmbedder
from hopiumcode.indexing.vectorstore import IndexNotBuilt, VectorIndex

from .fakes import FakeCollection, FakeEmbedder, FakeIndex

CONNECT = "def connect(addr):\n    try:\n        return socket.create_connection(addr)\n    except OSError as ex:\n        raise ConnectionFailed(ex)\n"
RENDER = "def render(template, context):\n    return template.format(**context)\n"


def make_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "net.py").write_text(CONNECT)
    (repo / "view.py").write_text(RENDER)
    (repo / "README.md").write_text("# Usage\nCall connect to open a socket.\n")
    return repo


def cfg(tmp_path, repo, model="nomic-embed-text"):
    return Config(repo_root=repo, index_dir=tmp_path / "index", graph_dir=tmp_path / "graph",
                  embed_model=model)


def test_vector_search_roundtrip(tmp_path):
    repo = make_repo(tmp_path)
    index = VectorIndex(cfg(tmp_path, repo), FakeEmbedder(), FakeCollection())
    chunks = [c for f in ("net.py", "view.py") for c in chunk_file(repo / f, repo)]
    index.add(chunks)
    hits = index.search("create socket connection error", 2)
    assert hits[0].chunk.id == "net.py::connect" and hits[0].via == "vector"
    assert 0 <= hits[0].score <= 1 and hits[0].chunk.text == chunks[0].text
    assert (hits[0].chunk.start_line, hits[0].chunk.end_line) == (1, 5)
    assert hits[0].chunk.name == "connect" and hits[0].chunk.lang == "python"
    index.delete_by_path("net.py")
    assert all(h.chunk.path != "net.py" for h in index.search("socket", 5))


def test_search_empty_index_raises(tmp_path):
    index = VectorIndex(cfg(tmp_path, tmp_path), FakeEmbedder(), FakeCollection())
    try:
        index.search("q", 3)
    except IndexNotBuilt as ex:
        assert "make index" in str(ex)
    else:
        raise AssertionError("ожидали IndexNotBuilt")


def test_embedding_text_has_path_and_name(tmp_path):
    repo = make_repo(tmp_path)
    emb = FakeEmbedder()
    VectorIndex(cfg(tmp_path, repo), emb, FakeCollection()).add(chunk_file(repo / "net.py", repo))
    assert emb.texts[0].startswith("# net.py :: connect\n")


class FakeClient:
    def __init__(self):
        self.calls = []

    def embed(self, model, input):
        self.calls.append(list(input))
        return {"embeddings": [[float(len(t)), 1.0] for t in input]}


def test_embedder_cache_and_batches(tmp_path):
    client = FakeClient()
    emb = OllamaEmbedder("bge-m3", cache_path=tmp_path / "c.sqlite", client=client)
    texts = [f"text {i}" for i in range(70)]
    first = emb.embed(texts)
    assert [len(c) for c in client.calls] == [32, 32, 6]          # пакеты по 32
    again = emb.embed(texts + ["new"])
    assert len(client.calls) == 4 and client.calls[-1] == ["new"]  # из кэша взято всё, кроме нового
    assert again[:70] == first
    emb.close()
    emb2 = OllamaEmbedder("bge-m3", cache_path=tmp_path / "c.sqlite", client=client)
    emb2.embed(texts)
    assert len(client.calls) == 4                                  # кэш пережил перезапуск


def test_embedder_retries_then_raises(tmp_path):
    class Flaky:
        n = 0

        def embed(self, model, input):
            self.n += 1
            if self.n < 3:
                raise ConnectionError("down")
            return {"embeddings": [[1.0] for _ in input]}

    flaky = Flaky()
    assert OllamaEmbedder(client=flaky, sleep=0).embed(["a"]) == [[1.0]] and flaky.n == 3

    class Dead:
        n = 0

        def embed(self, model, input):
            self.n += 1
            raise ConnectionError("down")

    dead = Dead()
    try:
        OllamaEmbedder(client=dead, sleep=0).embed(["a"])
    except ConnectionError:
        assert dead.n == 3
    else:
        raise AssertionError("ожидали ConnectionError")


def test_nomic_prefix_applies_only_to_nomic():
    client = FakeClient()
    OllamaEmbedder("nomic-embed-text", client=client).embed(["x"], kind="query")
    OllamaEmbedder("bge-m3", client=client).embed(["x"], kind="query")
    assert client.calls[0] == ["search_query: x"] and client.calls[1] == ["x"]


def test_build_is_incremental(tmp_path):
    repo = make_repo(tmp_path)
    c = cfg(tmp_path, repo)
    idx = FakeIndex()
    s1 = build_all(c, index=idx, graph=False)
    assert s1["files"] == 3 and s1["changed"] == 3 and idx.resets == 1
    assert "net.py::connect" in idx.added and "README.md::#0" in idx.added

    idx2 = FakeIndex()
    s2 = build_all(c, index=idx2, graph=False)                    # ничего не менялось
    assert s2["changed"] == 0 and idx2.added == [] and idx2.deleted == [] and idx2.resets == 0

    (repo / "view.py").write_text(RENDER + "\ndef extra():\n    pass\n")
    (repo / "README.md").unlink()
    (repo / "new.py").write_text("def fresh():\n    pass\n")
    idx3 = FakeIndex()
    s3 = build_all(c, index=idx3, graph=False)
    assert s3["changed"] == 2 and s3["removed"] == 1
    assert sorted(idx3.deleted) == ["README.md", "view.py"]
    assert "view.py::extra" in idx3.added and "new.py::fresh" in idx3.added
    assert not any(i.startswith("net.py") for i in idx3.added)    # нетронутый файл не переиндексируется
    manifest = json.loads((c.index_dir / "manifest.json").read_text())
    assert set(manifest["files"]) == {"net.py", "view.py", "new.py"}


def test_build_full_rebuild_on_model_change(tmp_path):
    repo = make_repo(tmp_path)
    build_all(cfg(tmp_path, repo), index=FakeIndex(), graph=False)
    idx = FakeIndex()
    stats = build_all(cfg(tmp_path, repo, model="bge-m3"), index=idx, graph=False)
    assert stats["changed"] == 3 and idx.resets == 1


def test_build_builds_graph(tmp_path):
    repo = make_repo(tmp_path)
    c = cfg(tmp_path, repo)
    stats = build_all(c, index=FakeIndex(), graph=True)
    assert stats["graph"]["nodes"]["Function"] >= 2
    assert (c.graph_dir / "graph.json").exists()


def test_build_missing_repo(tmp_path):
    try:
        build_all(cfg(tmp_path, tmp_path / "nope"), index=FakeIndex(), graph=False)
    except FileNotFoundError as ex:
        assert "make fetch" in str(ex)
    else:
        raise AssertionError("ожидали FileNotFoundError")
