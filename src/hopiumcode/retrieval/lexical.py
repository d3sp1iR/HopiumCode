"""BM25 по фрагментам и слияние выдач. Владелец: участник 2.

Lexical: BM25Okapi (rank_bm25) по тем же Chunk, что и в векторном индексе. Документ фрагмента: путь,
qualified name и текст. Токены: нижний регистр, идентификатор целиком и его части по "_" и camelCase
(parse_config -> parse_config, parse, config), не короче 2 символов, без русских и английских стоп-слов.
Lexical() без аргументов загружает индекс, сохранённый build_all в Config.index_dir / "bm25.json".

fuse: Reciprocal Rank Fusion нескольких выдач; via объединяется через "+", score нормализуется в 0..1.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

from ..config import Config
from ..models import Chunk, Hit

INDEX_FILE = "bm25.json"

STOP_WORDS = frozenset("""
a an and are as at be by for from has have if in into is it its of on or so than that the then this to
was were what when where which who why will with how does do can should would there their about
self cls def return import none true false pass elif else
и в во не что как на с со по к ко из за от до для о об при это этот эта эти то же ли где кто или а но
у мы вы он она оно они его её ее их есть был была были быть все всё так там тут ещё еще уже чем чего
какой какая какие каким какую когда если бы только можно нужно ли надо тот та те там при нет да
""".split())  # noqa: SIM905 - список читать проще строкой

_TOKEN = re.compile(r"[A-Za-zА-Яа-яЁё_][A-Za-zА-Яа-яЁё0-9_]*|\d+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[А-ЯЁ]?[а-яё]+|[А-ЯЁ]+|\d+")


def split_identifier(word: str) -> list[str]:
    """Части идентификатора по "_" и camelCase: SecureCookieSession -> Secure, Cookie, Session."""
    return [p for part in word.split("_") for p in _CAMEL.findall(part)]


def tokenize(text: str) -> list[str]:
    tokens = []
    for word in _TOKEN.findall(text):
        whole = word.strip("_").lower().replace("ё", "е")
        parts = [p.lower().replace("ё", "е") for p in split_identifier(word)]
        for tok in ([whole] if whole else []) + (parts if parts != [whole] else []):
            if len(tok) >= 2 and tok not in STOP_WORDS:
                tokens.append(tok)
    return tokens


def _document(c: Chunk) -> str:
    return f"{c.path} {c.name or ''}\n{c.text}"


class Lexical:
    def __init__(self, chunks: list[Chunk] | None = None, path: Path | None = None) -> None:
        self.chunks: list[Chunk] = []
        self._bm25 = None
        if chunks is not None:
            self.fit(chunks)
        else:
            file = Path(path) if path else Config.from_env().index_dir / INDEX_FILE
            if file.exists():
                self.fit(self._read(file))

    def fit(self, chunks: list[Chunk]) -> None:
        from rank_bm25 import BM25Okapi

        self.chunks = list(chunks)
        self._bm25 = BM25Okapi([tokenize(_document(c)) for c in self.chunks]) if self.chunks else None

    def search(self, query: str, k: int) -> list[Hit]:
        q = tokenize(query)
        if self._bm25 is None or not q or k <= 0:
            return []
        scores = self._bm25.get_scores(q)
        order = sorted((i for i, s in enumerate(scores) if s > 0), key=lambda i: (-scores[i], i))[:k]
        if not order:
            return []
        top = scores[order[0]]
        return [Hit(self.chunks[i], float(scores[i] / top), "bm25") for i in order]

    def save(self, path: Path | None = None) -> Path:
        file = Path(path) if path else Config.from_env().index_dir / INDEX_FILE
        file.parent.mkdir(parents=True, exist_ok=True)
        tmp = file.with_suffix(".tmp")
        tmp.write_text(json.dumps([asdict(c) for c in self.chunks], ensure_ascii=False), encoding="utf-8")
        tmp.replace(file)
        return file

    @staticmethod
    def _read(file: Path) -> list[Chunk]:
        return [Chunk(**d) for d in json.loads(file.read_text(encoding="utf-8"))]


def fuse(lists: list[list[Hit]], k: int = 60, top: int = 20) -> list[Hit]:
    """Reciprocal Rank Fusion; в via объединить источники через «+»."""
    scores: dict[str, float] = {}
    first: dict[str, Hit] = {}
    vias: dict[str, list[str]] = {}
    for hits in lists:
        seen: set[str] = set()
        for rank, h in enumerate(hits, start=1):
            cid = h.chunk.id
            if cid in seen:  # повтор внутри одной выдачи не усиливает фрагмент
                continue
            seen.add(cid)
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            first.setdefault(cid, h)
            for v in h.via.split("+"):
                if v and v not in vias.setdefault(cid, []):
                    vias[cid].append(v)
    order = list(first)  # при равенстве раньше тот, кто встретился раньше
    ranked = sorted(order, key=lambda cid: (-scores[cid], order.index(cid)))[:top]
    if not ranked:
        return []
    best = scores[ranked[0]]
    return [Hit(first[cid].chunk, scores[cid] / best, "+".join(vias[cid])) for cid in ranked]
