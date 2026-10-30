"""Эмбеддинги через Ollama с кэшем на диске. Владелец: участник 2."""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

from ..models import Chunk

log = logging.getLogger(__name__)

BATCH = 32
RETRIES = 3

# Модели nomic-embed-text обучены с префиксами задач (см. карточку модели).
_NOMIC_PREFIX = {"document": "search_document: ", "query": "search_query: "}


def chunk_embed_text(chunk: Chunk) -> str:
    """Текст для эмбеддинга: заголовок «путь + имя символа» и тело фрагмента."""
    return f"# {chunk.path} :: {chunk.name or chunk.kind}\n{chunk.text}"


class OllamaEmbedder:
    """embed(texts) -> векторы. Пакеты по 32, 3 повтора при ошибке, кэш по sha256 текста."""

    def __init__(self, model: str = "nomic-embed-text", host: str | None = None,
                 cache_path: Path | str | None = None, client: Any | None = None,
                 sleep: float = 1.0) -> None:
        self.model = model
        self.host = host
        self.sleep = sleep
        self._client = client
        self._db: sqlite3.Connection | None = None
        if cache_path is not None:
            path = Path(cache_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(path)
            self._db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec TEXT NOT NULL)")
        self.calls = 0   # сколько пакетов реально ушло в Ollama (для статистики и тестов)

    # --- кэш
    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}\0{text}".encode()).hexdigest()

    def _cache_get(self, keys: list[str]) -> dict[str, list[float]]:
        if self._db is None or not keys:
            return {}
        found: dict[str, list[float]] = {}
        for i in range(0, len(keys), 500):
            part = keys[i:i + 500]
            marks = ",".join("?" * len(part))
            for key, vec in self._db.execute(f"SELECT key, vec FROM emb WHERE key IN ({marks})", part):
                found[key] = json.loads(vec)
        return found

    def _cache_put(self, items: dict[str, list[float]]) -> None:
        if self._db is None or not items:
            return
        self._db.executemany("INSERT OR REPLACE INTO emb (key, vec) VALUES (?, ?)",
                             [(k, json.dumps(v)) for k, v in items.items()])
        self._db.commit()

    # --- Ollama
    def _get_client(self) -> Any:
        if self._client is None:
            import ollama
            self._client = ollama.Client(host=self.host) if self.host else ollama.Client()
        return self._client

    def _request(self, batch: list[str]) -> list[list[float]]:
        last: Exception | None = None
        for attempt in range(RETRIES):
            try:
                self.calls += 1
                resp = self._get_client().embed(model=self.model, input=batch)
                vectors = resp["embeddings"] if isinstance(resp, dict) else resp.embeddings
                if len(vectors) != len(batch):
                    raise RuntimeError(f"Ollama вернула {len(vectors)} векторов на {len(batch)} текстов")
                return [list(map(float, v)) for v in vectors]
            except Exception as exc:  # noqa: BLE001 - повторяем любую ошибку сети и модели
                last = exc
                log.warning("эмбеддинги: попытка %d из %d не удалась: %s", attempt + 1, RETRIES, exc)
                if attempt + 1 < RETRIES:
                    time.sleep(self.sleep)
        assert last is not None
        raise last

    def embed(self, texts: list[str], kind: str = "document") -> list[list[float]]:
        """kind: "document" для фрагментов, "query" для вопроса (важно для nomic-embed-text)."""
        prefix = _NOMIC_PREFIX.get(kind, "") if self.model.startswith("nomic-embed") else ""
        prepared = [prefix + t for t in texts]
        keys = [self._key(t) for t in prepared]
        cached = self._cache_get(list(dict.fromkeys(keys)))
        missing = list(dict.fromkeys(k for k in keys if k not in cached))
        by_key = dict(zip(keys, prepared))
        for i in range(0, len(missing), BATCH):
            part = missing[i:i + BATCH]
            vectors = self._request([by_key[k] for k in part])
            fresh = dict(zip(part, vectors))
            self._cache_put(fresh)
            cached.update(fresh)
        return [cached[k] for k in keys]

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None
