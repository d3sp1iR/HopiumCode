"""Запросы к графу и обход"""
from __future__ import annotations

from ..config import Config
from .store import GraphStore, Resolution, open_store

EDGE_WEIGHT = {"CALLS": 0.7, "INHERITS": 0.7, "DOCUMENTS": 0.6, "IMPORTS": 0.4, "DEFINES": 0.5}
DEGREE_PENALTY = 0.15
THRESHOLD = 0.05

_store: GraphStore | None = None


def get_store(config: Config | None = None) -> GraphStore:
    global _store
    if _store is None:
        store = open_store(config)
        store.load()
        _store = store
    return _store


def resolve(symbol: str, store: GraphStore | None = None) -> Resolution:
    return (store or get_store()).resolve(symbol)


def callers(symbol: str, depth: int = 1, store: GraphStore | None = None) -> list[str]:
    return (store or get_store()).callers(symbol, depth)


def callees(symbol: str, depth: int = 1, store: GraphStore | None = None) -> list[str]:
    return (store or get_store()).callees(symbol, depth)


def expand(
    seed_ids: list[str],
    seed_scores: dict[str, float],
    hops: int = 2,
    store: GraphStore | None = None,
) -> list[tuple[str, float, str]]:
    """Вес соседа = вес родителя * w(тип ребра) / (1 + 0.15 * (степень - 1)). Вернуть (id, вес, причина)."""
    store = store or get_store()
    seeds = set(seed_ids)
    best: dict[str, tuple[float, str]] = {}
    frontier = {s: seed_scores.get(s, 1.0) for s in seed_ids}
    for _ in range(hops):
        nxt: dict[str, float] = {}
        for node, weight in frontier.items():
            for other, etype in store.neighbors(node):
                if other in seeds:
                    continue
                w = weight * EDGE_WEIGHT.get(etype, 0.0)
                w /= 1 + DEGREE_PENALTY * (store.degree(other) - 1)
                if w < THRESHOLD or w <= best.get(other, (0.0, ""))[0]:
                    continue
                best[other] = (w, f"graph:{etype.lower()}")
                nxt[other] = w
        frontier = nxt
    return sorted(((n, w, via) for n, (w, via) in best.items()), key=lambda t: -t[1])
