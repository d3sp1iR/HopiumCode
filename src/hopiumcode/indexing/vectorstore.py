"""Векторная БД (Chroma). Владелец: участник 2.

VectorIndex хранит фрагменты в persistent-коллекции Chroma (по умолчанию Config.index_dir / "chroma").
Векторы считает переданный эмбеддер (по умолчанию OllamaEmbedder): Chroma свою модель не загружает.
Метаданные: path, start_line, end_line, kind, name, lang; текст фрагмента лежит в documents, поэтому
Hit.chunk восстанавливается полностью. Метрика косинусная, score = 1 - distance / 2 (0..1, больше = ближе).
"""
from __future__ import annotations

from pathlib import Path

from ..config import Config
from ..models import Chunk, Hit
from .embed import Embedder, embedding_text

_ADD_BATCH = 1000


def _meta(c: Chunk) -> dict:
    return {"path": c.path, "start_line": c.start_line, "end_line": c.end_line, "kind": c.kind,
            "name": c.name or "", "lang": c.lang}  # Chroma не хранит None


def _chunk(cid: str, doc: str, m: dict) -> Chunk:
    return Chunk(cid, m["path"], int(m["start_line"]), int(m["end_line"]), m["kind"], m["name"] or None, doc,
                 m["lang"])


class VectorIndex:
    def __init__(self, path: Path | None = None, embedder: Embedder | None = None,
                 collection: str = "chunks") -> None:
        import chromadb
        from chromadb.config import Settings

        cfg = Config.from_env()
        self.path = Path(path or cfg.index_dir / "chroma")
        self.path.mkdir(parents=True, exist_ok=True)
        if embedder is None:
            from .embed import OllamaEmbedder
            embedder = OllamaEmbedder()
        self.embedder = embedder
        self._client = chromadb.PersistentClient(path=str(self.path), settings=Settings(anonymized_telemetry=False))
        self._col = self._client.get_or_create_collection(collection, embedding_function=None,
                                                          metadata={"hnsw:space": "cosine"})

    def add(self, chunks: list[Chunk]) -> None:
        """Добавляет или заменяет фрагменты (по id)."""
        for i in range(0, len(chunks), _ADD_BATCH):
            part = chunks[i:i + _ADD_BATCH]
            vectors = self.embedder.embed([embedding_text(c) for c in part])
            self._col.upsert(ids=[c.id for c in part], embeddings=vectors, documents=[c.text for c in part],
                             metadatas=[_meta(c) for c in part])

    def search(self, query: str, k: int) -> list[Hit]:
        n = self.count()
        if n == 0 or k <= 0:
            return []
        vec = self.embedder.embed([query])[0]
        res = self._col.query(query_embeddings=[vec], n_results=min(k, n),
                              include=["documents", "metadatas", "distances"])
        hits = []
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0],
                                        res["distances"][0]):
            score = min(1.0, max(0.0, 1.0 - float(dist) / 2.0))
            hits.append(Hit(_chunk(cid, doc, meta), score, "vector"))
        return hits

    def delete_by_path(self, path: str) -> None:
        self._col.delete(where={"path": path})

    def count(self) -> int:
        return self._col.count()

    def get_all(self) -> list[Chunk]:
        """Все фрагменты индекса (например, чтобы построить BM25 по тем же Chunk)."""
        res = self._col.get(include=["documents", "metadatas"])
        chunks = [_chunk(cid, doc, m) for cid, doc, m in zip(res["ids"], res["documents"], res["metadatas"])]
        return sorted(chunks, key=lambda c: (c.path, c.start_line, c.id))
