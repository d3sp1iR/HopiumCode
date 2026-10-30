"""BM25 по фрагментам и слияние выдач. Владелец: участник 2."""
from __future__ import annotations

import math
import re
from collections import Counter

from ..models import Chunk, Hit

_WORD = re.compile(r"[^\W_]+(?:_+[^\W_]+)*", re.UNICODE)    # слово или идентификатор с подчёркиваниями
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+|[^\W\d_]+", re.UNICODE)

STOPWORDS = frozenset("""
a an and are as at be by for from has have in is it its of on or that the this to was were will with
if not no do does how what where when why who which can
py md rst txt self cls
и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было
вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас
нибудь опять уж вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их
чем была сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой
совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех никогда можно при
наконец два об другой хоть после над больше тот через эти нас про них какая много разве три эту
моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им более всегда конечно всю
между как какие какой какая каком где чем это эта эти этот
""".split())


def split_identifier(word: str) -> list[str]:
    """parse_config -> [parse, config]; parseConfig -> [parse, config]; HTTPError -> [http, error]."""
    parts: list[str] = []
    for piece in word.split("_"):
        parts.extend(m.group(0).lower() for m in _CAMEL.finditer(piece))
    return parts


def tokenize(text: str) -> list[str]:
    """Нижний регистр, разбиение идентификаторов, слова от 2 символов, без стоп-слов.

    Для составного идентификатора сохраняется и целая форма (parse_config), и части.
    """
    out: list[str] = []
    for word in _WORD.findall(text):
        parts = split_identifier(word)
        tokens = list(parts)
        if len(parts) > 1:
            tokens.append(word.lower())
        out.extend(t for t in tokens if len(t) >= 2 and t not in STOPWORDS)
    return out


class _BM25:
    """BM25 Okapi (запасная реализация, если rank_bm25 не установлен)."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tf = [Counter(doc) for doc in corpus]
        self.len = [len(doc) for doc in corpus]
        self.avg = (sum(self.len) / len(corpus)) if corpus else 0.0
        df: Counter[str] = Counter()
        for tf in self.tf:
            df.update(tf.keys())
        n = len(corpus)
        self.idf = {t: math.log(1.0 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def get_scores(self, query: list[str]) -> list[float]:
        scores = [0.0] * len(self.tf)
        for term in set(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(term, 0)
                if f:
                    norm = 1 - self.b + self.b * self.len[i] / (self.avg or 1.0)
                    scores[i] += idf * f * (self.k1 + 1) / (f + self.k1 * norm)
        return scores


def _make_bm25(corpus: list[list[str]]):
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return _BM25(corpus)
    return BM25Okapi(corpus, k1=1.5, b=0.75)


class Lexical:
    """BM25 по тем же фрагментам, что и в векторном индексе. score нормируется в 0..1."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = list(chunks)
        corpus = [tokenize(f"{c.path} {c.name or ''} {c.text}") for c in self.chunks]
        self._bm25 = _make_bm25(corpus) if self.chunks else None

    def search(self, query: str, k: int) -> list[Hit]:
        tokens = tokenize(query)
        if not tokens or self._bm25 is None:
            return []
        scores = list(self._bm25.get_scores(tokens))
        order = sorted((i for i, s in enumerate(scores) if s > 0), key=lambda i: -scores[i])[:k]
        if not order:
            return []
        top = scores[order[0]]
        return [Hit(self.chunks[i], scores[i] / top, "bm25") for i in order]


def fuse(lists: list[list[Hit]], k: int = 60, top: int = 20) -> list[Hit]:
    """Reciprocal Rank Fusion; в via источники объединяются через «+».

    Оценка фрагмента = сумма 1/(k + место) по спискам, где он найден; затем делится на максимум
    (len(lists)/(k+1)), поэтому лежит в 0..1. Возвращается Hit с самым высоким исходным score.
    """
    if not lists:
        return []
    rrf: dict[str, float] = {}
    best: dict[str, Hit] = {}
    via: dict[str, list[str]] = {}
    for hits in lists:
        seen: set[str] = set()
        rank = 0
        for hit in hits:
            if hit.chunk.id in seen:
                continue
            seen.add(hit.chunk.id)
            rank += 1
            cid = hit.chunk.id
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in best or hit.score > best[cid].score:
                best[cid] = hit
            for source in hit.via.split("+"):
                if source not in via.setdefault(cid, []):
                    via[cid].append(source)
    norm = len(lists) / (k + 1)
    order = sorted(rrf, key=lambda cid: -rrf[cid])      # sorted устойчива: при равенстве порядок появления
    return [Hit(best[cid].chunk, min(1.0, rrf[cid] / norm), "+".join(via[cid])) for cid in order[:top]]
