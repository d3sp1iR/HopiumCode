"""Векторная БД (Chroma). Владелец: участник 2."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..config import Config
from ..models import Chunk, Hit
from .embed import OllamaEmbedder, chunk_embed_text

log = logging.getLogger(__name__)

COLLECTION = "chunks"
ADD_BATCH = 128


class IndexNotBuilt(RuntimeError):
    """Векторный индекс пуст или не создан."""


class VectorIndex:
    """add / search / delete_by_path поверх Chroma (persistent client в index_dir/chroma).

    Конструктор без аргументов берёт настройки из окружения (так его вызывает retrieval.hybrid).
    embedder и collection можно подменить в тестах.
    """

    def __init__(self, config: Config | None = None, embedder: OllamaEmbedder | None = None,
                 collection: Any | None = None) -> None:
        self.config = config or Config.from_env()
        self.embedder = embedder or OllamaEmbedder(
            self.config.embed_model, self.config.ollama_url,
            cache_path=self.config.index_dir / "embed_cache.sqlite")
        self._collection = collection
        self._client: Any | None = None

    def _chroma(self) -> Any:
        if self._client is None:
            import chromadb
            path = Path(self.config.index_dir) / "chroma"
            path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(path))
        return self._client

    @property
    def collection(self) -> Any:
        if self._collection is None:
            self._collection = self._chroma().get_or_create_collection(
                COLLECTION, metadata={"hnsw:space": "cosine"})
        return self._collection

    def reset(self) -> None:
        """Удаляет коллекцию целиком (нужно при смене модели эмбеддингов: другая размерность)."""
        try:
            self._chroma().delete_collection(COLLECTION)
        except Exception as exc:  # noqa: BLE001 - коллекции могло не быть
            log.debug("коллекция %s не удалена: %s", COLLECTION, exc)
        self._collection = None

    def count(self) -> int:
        return int(self.collection.count())

    def add(self, chunks: list[Chunk]) -> None:
        for i in range(0, len(chunks), ADD_BATCH):
            part = chunks[i:i + ADD_BATCH]
            vectors = self.embedder.embed([chunk_embed_text(c) for c in part], kind="document")
            self.collection.upsert(
                ids=[c.id for c in part],
                embeddings=vectors,
                documents=[c.text for c in part],
                metadatas=[{"path": c.path, "start_line": c.start_line, "end_line": c.end_line,
                            "kind": c.kind, "name": c.name or "", "lang": c.lang} for c in part])

    def search(self, query: str, k: int) -> list[Hit]:
        total = self.count()
        if total == 0:
            raise IndexNotBuilt("Векторный индекс пуст: выполните `make index`")
        vector = self.embedder.embed([query], kind="query")[0]
        res = self.collection.query(query_embeddings=[vector], n_results=min(k, total))
        hits: list[Hit] = []
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                        res["metadatas"][0], res["distances"][0]):
            chunk = Chunk(cid, meta["path"], int(meta["start_line"]), int(meta["end_line"]),
                          meta["kind"], meta["name"] or None, doc, meta["lang"])
            hits.append(Hit(chunk, max(0.0, min(1.0, 1.0 - float(dist))), "vector"))
        return hits

    def delete_by_path(self, path: str) -> None:
        self.collection.delete(where={"path": path})
