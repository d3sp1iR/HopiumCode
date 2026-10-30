"""Гибридный ретривер: вектор + BM25 + обход графа"""
from __future__ import annotations

import dataclasses
from collections.abc import Callable

from ..config import Config
from ..graph.query import expand
from ..graph.store import GraphStore, open_store
from ..interfaces import LexicalLike, VectorIndexLike
from ..models import Chunk, Hit

MODES = ("vector", "vector+bm25", "hybrid")


def _rescale(hits: list[Hit]) -> list[Hit]:
    if not hits:
        return []
    scores = [h.score for h in hits]
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return [dataclasses.replace(h, score=1.0) for h in hits]
    return [dataclasses.replace(h, score=(h.score - lo) / (hi - lo)) for h in hits]


def _merge(*groups: list[Hit]) -> list[Hit]:
    best: dict[str, Hit] = {}
    for hit in (h for group in groups for h in group):
        cur = best.get(hit.chunk.id)
        if cur is None:
            best[hit.chunk.id] = hit
            continue
        via = "+".join(dict.fromkeys(cur.via.split("+") + hit.via.split("+")))
        winner = cur if cur.score >= hit.score else hit
        best[hit.chunk.id] = dataclasses.replace(winner, via=via)
    return sorted(best.values(), key=lambda h: -h.score)


class Retriever:
    def __init__(
        self,
        vector: VectorIndexLike,
        lexical: LexicalLike,
        graph: GraphStore,
        chunks: dict[str, Chunk],
        max_chars: int = 4000,
        pool: int = 20,
        seeds: int = 3,
        hops: int = 2,
        fuse: Callable[..., list[Hit]] | None = None,
        graph_quota: int = 2,
        expand_query: Callable[[str], str] | None = None,
    ) -> None:
        if fuse is None:
            from .lexical import fuse
        self.vector, self.lexical, self.graph, self.chunks = vector, lexical, graph, chunks
        self.max_chars, self.pool, self.seeds, self.hops = max_chars, pool, seeds, hops
        self.fuse = fuse
        self.graph_quota = graph_quota          # мест в выдаче hybrid под фрагменты, найденные только графом
        self.expand_query = expand_query or (lambda q: q)

    def retrieve(self, question: str, k: int = 8, mode: str = "hybrid") -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"неизвестный режим: {mode}")
        query = self.expand_query(question)
        vec = self.vector.search(query, self.pool)
        if mode == "vector":
            return self._pack(vec, k)
        fused = self.fuse([vec, self.lexical.search(query, self.pool)], top=self.pool)
        if mode == "vector+bm25":
            return self._pack(fused, k)
        base = _rescale(fused)
        graph_hits = self._graph_hits(base)
        # Граф добавляет связи (вызывающие, наследники), но их веса после затухания ниже, чем у прямых
        # совпадений, и без квоты они не попадали в выдачу. Лучшие фрагменты, которых нет в выдаче поиска,
        # получают graph_quota гарантированных мест.
        # Первыми резервируются те, кто вызывает лучший результат: их текстовый поиск не найдёт никогда.
        # Остальные места квоты: находки графа, которых нет среди верхних результатов поиска.
        quota = min(self.graph_quota, k - 1)
        graph_score = {h.chunk.id: h for h in graph_hits}
        top_callers = self.graph.related(base[0].chunk.id, "CALLS", True) if base else []
        reserved = [graph_score.get(c) or Hit(self.chunks[c], 0.5, "graph:calls")
                    for c in top_callers if c in self.chunks][:quota]
        taken = {h.chunk.id for h in reserved} | {h.chunk.id for h in base[: max(1, k - quota)]}
        reserved += [h for h in graph_hits if h.chunk.id not in taken][: quota - len(reserved)]
        budget = self.max_chars - sum(len(h.chunk.text) for h in reserved)
        reserved_ids = {h.chunk.id for h in reserved}
        main = self._pack([h for h in _merge(base, graph_hits) if h.chunk.id not in reserved_ids],
                          k - len(reserved), budget)
        return main + reserved

    def _graph_hits(self, base: list[Hit]) -> list[Hit]:
        top = base[: self.seeds]
        scores = {h.chunk.id: h.score for h in top}
        hits = []
        for node_id, weight, via in expand(list(scores), scores, self.hops, store=self.graph):
            chunk = self.chunks.get(node_id)
            if chunk:
                hits.append(Hit(chunk, weight, via))
        return hits

    def _pack(self, hits: list[Hit], k: int, max_chars: int | None = None) -> list[Hit]:
        limit = self.max_chars if max_chars is None else max_chars
        out: list[Hit] = []
        seen: set[str] = set()
        used = 0
        for hit in hits:
            if hit.chunk.id in seen:
                continue
            size = len(hit.chunk.text)
            if out and used + size > limit:
                continue
            out.append(hit)
            seen.add(hit.chunk.id)
            used += size
            if len(out) == k:
                break
        return out


def build_default(config: Config | None = None) -> Retriever:
    from ..indexing.chunker import chunk_file
    from ..indexing.loader import iter_files
    from ..indexing.vectorstore import VectorIndex
    from .lexical import Lexical

    cfg = config or Config.from_env()
    chunks = [c for p in iter_files(cfg.repo_root) for c in chunk_file(p, cfg.repo_root)]
    graph = open_store(cfg)
    graph.load()
    from . import query_expansion

    expand = query_expansion.expand_query if query_expansion.enabled() else None
    return Retriever(VectorIndex(), Lexical(chunks), graph, {c.id: c for c in chunks}, expand_query=expand)


_default: Retriever | None = None


def retrieve(question: str, k: int = 8, mode: str = "hybrid") -> list[Hit]:
    """mode: "vector" | "vector+bm25" | "hybrid" (с графом). Режимы нужны для сравнения в eval."""
    global _default
    if _default is None:
        _default = build_default()
    return _default.retrieve(question, k, mode)