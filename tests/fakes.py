"""Подставные эмбеддинги и коллекция Chroma для тестов индексации без Ollama и chromadb."""
from __future__ import annotations

import hashlib
import math

from hopiumcode.retrieval.lexical import tokenize

DIM = 64


def vec(text: str) -> list[float]:
    """Детерминированный «мешок слов» в 64 измерениях, нормированный."""
    v = [0.0] * DIM
    for tok in tokenize(text):
        v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


class FakeEmbedder:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed(self, texts: list[str], kind: str = "document") -> list[list[float]]:
        self.texts.extend(texts)
        return [vec(t) for t in texts]


class FakeCollection:
    """Минимальное подмножество API Chroma, которое использует VectorIndex."""

    def __init__(self) -> None:
        self.rows: dict[str, tuple[list[float], str, dict]] = {}

    def count(self) -> int:
        return len(self.rows)

    def upsert(self, ids, embeddings, documents, metadatas) -> None:
        for i, e, d, m in zip(ids, embeddings, documents, metadatas):
            self.rows[i] = (e, d, m)

    def delete(self, where) -> None:
        (key, value), = where.items()
        self.rows = {i: r for i, r in self.rows.items() if r[2].get(key) != value}

    def query(self, query_embeddings, n_results) -> dict:
        q = query_embeddings[0]
        scored = sorted(((1.0 - sum(a * b for a, b in zip(q, e)), i) for i, (e, _, _) in self.rows.items()))
        top = scored[:n_results]
        return {"ids": [[i for _, i in top]],
                "documents": [[self.rows[i][1] for _, i in top]],
                "metadatas": [[self.rows[i][2] for _, i in top]],
                "distances": [[d for d, _ in top]]}


class FakeIndex:
    """Индекс-заглушка для build_all: запоминает вызовы."""

    def __init__(self) -> None:
        self.added: list[str] = []
        self.deleted: list[str] = []
        self.resets = 0

    def add(self, chunks) -> None:
        self.added.extend(c.id for c in chunks)

    def delete_by_path(self, path: str) -> None:
        self.deleted.append(path)

    def reset(self) -> None:
        self.resets += 1
