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
    ) -> None:
        if fuse is None:
            from .lexical import fuse
        self.vector, self.lexical, self.graph, self.chunks = vector, lexical, graph, chunks
        self.max_chars, self.pool, self.seeds, self.hops = max_chars, pool, seeds, hops
        self.fuse = fuse

    def retrieve(self, question: str, k: int = 8, mode: str = "hybrid") -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"неизвестный режим: {mode}")
        vec = self.vector.search(question, self.pool)
        if mode == "vector":
            return self._pack(vec, k)
        fused = self.fuse([vec, self.lexical.search(question, self.pool)], top=self.pool)
        if mode == "vector+bm25":
            return self._pack(fused, k)
        base = _rescale(fused)
        return self._pack(_merge(base, self._graph_hits(base)), k)

    def _graph_hits(self, base: list[Hit]) -> list[Hit]:
        top = base[: self.seeds]
        scores = {h.chunk.id: h.score for h in top}
        hits = []
        for node_id, weight, via in expand(list(scores), scores, self.hops, store=self.graph):
            chunk = self.chunks.get(node_id)
            if chunk:
                hits.append(Hit(chunk, weight, via))
        return hits

    def _pack(self, hits: list[Hit], k: int) -> list[Hit]:
        out: list[Hit] = []
        seen: set[str] = set()
        used = 0
        for hit in hits:
            if hit.chunk.id in seen:
                continue
            size = len(hit.chunk.text)
            if out and used + size > self.max_chars:
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
    return Retriever(VectorIndex(), Lexical(chunks), graph, {c.id: c for c in chunks})


_default: Retriever | None = None


def retrieve(question: str, k: int = 8, mode: str = "hybrid") -> list[Hit]:
    """mode: "vector" | "vector+bm25" | "hybrid" (с графом). Режимы нужны для сравнения в eval."""
    global _default
    if _default is None:
        _default = build_default()
    return _default.retrieve(question, k, mode)