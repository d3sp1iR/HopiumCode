"""Эмбеддинги через Ollama с кэшем. Владелец: участник 2.

OllamaEmbedder: пакеты по 32 текста, до 3 попыток на пакет, кэш на диске (sqlite) по sha256 модели и текста.
Ошибки Ollama превращаются в EmbeddingError с понятным текстом и подсказкой, без трассировки.
HashEmbedder: детерминированный эмбеддер без сети для тестов и офлайн-проверок (качество не оценивает).
Любой объект с методом embed(texts) -> list[list[float]] подходит как эмбеддер (Embedder).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from ..config import Config
from ..models import Chunk

BATCH_SIZE = 32
RETRIES = 3


class EmbeddingError(RuntimeError):
    """Эмбеддинги получить не удалось; текст сообщения можно показывать пользователю как есть."""


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


def embedding_text(chunk: Chunk) -> str:
    """Текст для эмбеддинга фрагмента: заголовок "# путь :: имя" и код."""
    return f"# {chunk.path} :: {chunk.name or 'module'}\n{chunk.text}"


class EmbeddingCache:
    """Кэш векторов в sqlite: ключ sha256(модель + текст), значение JSON-список."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path)
        self._db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec TEXT NOT NULL)")

    @staticmethod
    def key(model: str, text: str) -> str:
        return hashlib.sha256(f"{model}\0{text}".encode()).hexdigest()

    def get_many(self, keys: list[str]) -> dict[str, list[float]]:
        found: dict[str, list[float]] = {}
        for i in range(0, len(keys), 500):  # лимит параметров sqlite
            part = keys[i:i + 500]
            rows = self._db.execute(f"SELECT key, vec FROM emb WHERE key IN ({','.join('?' * len(part))})", part)
            found.update((k, json.loads(v)) for k, v in rows)
        return found

    def put_many(self, items: dict[str, list[float]]) -> None:
        with self._db:
            self._db.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)",
                                 [(k, json.dumps(v)) for k, v in items.items()])

    def close(self) -> None:
        self._db.close()


def _describe(ex: Exception, model: str, host: str) -> str:
    status = getattr(ex, "status_code", None)
    text = str(getattr(ex, "error", "") or ex)
    if isinstance(ex, ConnectionError):
        return f"Ollama недоступна ({host}). Запустите `ollama serve` и повторите."
    if status == 404 or "not found" in text.lower():
        return f"Модель эмбеддингов {model} не скачана. Выполните: ollama pull {model}"
    return f"Ollama вернула ошибку для модели {model}: {text}"


class OllamaEmbedder:
    def __init__(self, model: str | None = None, *, host: str | None = None, cache_path: Path | None = None,
                 use_cache: bool = True, batch_size: int = BATCH_SIZE, retries: int = RETRIES, client=None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        cfg = Config.from_env()
        self.model = model or cfg.embed_model
        self.host = host or cfg.ollama_url
        self.batch_size = batch_size
        self.retries = retries
        self._client = client
        self._sleep = sleep
        self.cache = EmbeddingCache(cache_path or cfg.index_dir / "embed_cache.sqlite") if use_cache else None
        self.calls = 0          # запросов к Ollama (для статистики и тестов)
        self.cache_hits = 0

    def _get_client(self):
        if self._client is None:
            import ollama  # импорт здесь, чтобы тесты работали без пакета
            self._client = ollama.Client(host=self.host)
        return self._client

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        last: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                self.calls += 1
                resp = self._get_client().embed(model=self.model, input=batch)
                vectors = [list(map(float, v)) for v in (getattr(resp, "embeddings", None) or resp["embeddings"])]
                if len(vectors) != len(batch):
                    raise ValueError(f"получено {len(vectors)} векторов на {len(batch)} текстов")
                return vectors
            except Exception as ex:  # noqa: BLE001 - любая ошибка Ollama: повтор, затем понятное сообщение
                last = ex
                if getattr(ex, "status_code", None) == 404:  # модели нет: повтор не поможет
                    break
                if attempt < self.retries:
                    self._sleep(0.5 * attempt)
        raise EmbeddingError(_describe(last, self.model, self.host)) from None

    def embed(self, texts: list[str]) -> list[list[float]]:
        keys = [EmbeddingCache.key(self.model, t) for t in texts]
        known = self.cache.get_many(list(set(keys))) if self.cache else {}
        self.cache_hits += sum(k in known for k in keys)
        todo = list(dict.fromkeys((k, t) for k, t in zip(keys, texts) if k not in known))
        for i in range(0, len(todo), self.batch_size):
            part = todo[i:i + self.batch_size]
            new = dict(zip([k for k, _ in part], self._embed_batch([t for _, t in part])))
            known.update(new)
            if self.cache:
                self.cache.put_many(new)
        return [known[k] for k in keys]


_WORD = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")


class HashEmbedder:
    """Детерминированный эмбеддер без сети: мешок слов (и частей идентификаторов), хэшированный в dim измерений.

    Для тестов и офлайн-проверок цепочки; семантику не понимает.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.calls = 0

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in _WORD.findall(text.lower().replace("_", " ")):
            h = int.from_bytes(hashlib.md5(word.encode()).digest()[:4], "little")
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [self._vector(t) for t in texts]
