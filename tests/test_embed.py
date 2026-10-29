"""indexing.embed: пакеты, повторы, кэш, понятные ошибки. Без Ollama: клиент подменяется."""
import math

import pytest

from hopiumcode.indexing.embed import EmbeddingError, HashEmbedder, OllamaEmbedder, embedding_text
from hopiumcode.models import Chunk


class Resp:
    def __init__(self, embeddings):
        self.embeddings = embeddings


class FakeClient:
    """Возвращает вектор [len(text), номер вызова]; может падать заданное число раз."""

    def __init__(self, fail_times=0, error=None):
        self.batches, self.fail_times, self.error = [], fail_times, error

    def embed(self, model, input):
        self.batches.append(list(input))
        if self.fail_times:
            self.fail_times -= 1
            raise self.error or RuntimeError("временный сбой")
        return Resp([[float(len(t)), float(len(self.batches))] for t in input])


class NotFound(Exception):
    status_code = 404
    error = "model 'nomic-embed-text' not found"


def emb(tmp_path, client, **kw):
    return OllamaEmbedder("nomic-embed-text", client=client, cache_path=tmp_path / "cache.sqlite",
                          sleep=lambda s: None, **kw)


def test_batches_of_32_and_order(tmp_path):
    client = FakeClient()
    texts = [f"text {i:03d}" + "x" * i for i in range(70)]
    vecs = emb(tmp_path, client).embed(texts)
    assert [len(b) for b in client.batches] == [32, 32, 6]
    assert [v[0] for v in vecs] == [float(len(t)) for t in texts]


def test_cache_on_disk_survives_new_instance(tmp_path):
    first = FakeClient()
    emb(tmp_path, first).embed(["a", "bb"])
    second = FakeClient()
    e = emb(tmp_path, second)
    assert e.embed(["bb", "a", "ccc"]) == [[2.0, 1.0], [1.0, 1.0], [3.0, 1.0]]
    assert second.batches == [["ccc"]] and e.cache_hits == 2


def test_cache_key_depends_on_model(tmp_path):
    emb(tmp_path, FakeClient()).embed(["a"])
    other = FakeClient()
    OllamaEmbedder("other-model", client=other, cache_path=tmp_path / "cache.sqlite").embed(["a"])
    assert other.batches == [["a"]]


def test_duplicates_embedded_once(tmp_path):
    client = FakeClient()
    vecs = emb(tmp_path, client, use_cache=False).embed(["same", "same", "other"])
    assert client.batches == [["same", "other"]] and vecs[0] == vecs[1]


def test_retry_then_success(tmp_path):
    client = FakeClient(fail_times=2)
    assert emb(tmp_path, client).embed(["a"])[0][0] == 1.0
    assert len(client.batches) == 3


def test_gives_up_after_3_attempts(tmp_path):
    client = FakeClient(fail_times=5)
    with pytest.raises(EmbeddingError, match="ошибку"):
        emb(tmp_path, client).embed(["a"])
    assert len(client.batches) == 3


def test_connection_error_message(tmp_path):
    client = FakeClient(fail_times=5, error=ConnectionError("Failed to connect to Ollama"))
    with pytest.raises(EmbeddingError) as ei:
        emb(tmp_path, client).embed(["a"])
    assert "ollama serve" in str(ei.value) and ei.value.__cause__ is None and ei.value.__suppress_context__


def test_model_not_found_no_retry(tmp_path):
    client = FakeClient(fail_times=5, error=NotFound())
    with pytest.raises(EmbeddingError, match="ollama pull nomic-embed-text"):
        emb(tmp_path, client).embed(["a"])
    assert len(client.batches) == 1


def test_model_from_config(monkeypatch, tmp_path):
    monkeypatch.setenv("CG_EMBED_MODEL", "my-embed")
    assert OllamaEmbedder(use_cache=False).model == "my-embed"


def test_embedding_text_header():
    c = Chunk("app.py::Flask.run", "app.py", 1, 2, "method", "Flask.run", "def run(self): ...", "python")
    assert embedding_text(c) == "# app.py :: Flask.run\ndef run(self): ..."
    m = Chunk("app.py::#1", "app.py", 1, 2, "module", None, "import os", "python")
    assert embedding_text(m).startswith("# app.py :: module\n")


def test_hash_embedder_deterministic_and_normalized():
    e = HashEmbedder(dim=64)
    a, b = e.embed(["parse_config value", "parse config value"])
    assert a == b                                               # "_" разбивает идентификатор
    assert math.isclose(sum(x * x for x in a), 1.0)
    assert e.embed(["parse_config value"])[0] == a
