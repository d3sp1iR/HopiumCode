"""
Запросы к графу: callers, callees, subclasses, imports_of, docs_for
и обход expand() для гибридного ретривера (см. П3.3 ТЗ).

expand() реализует затухание веса и нормировку на степень узла:
вес соседа = вес родителя * w(тип ребра) / (1 + 0.15 * (степень - 1)).
"""
from __future__ import annotations

from dataclasses import dataclass

from codeguide.graph.store import GraphStore

# веса типов рёбер из ТЗ
EDGE_WEIGHTS: dict[str, float] = {
    "CALLS": 0.7,
    "INHERITS": 0.7,
    "DOCUMENTS": 0.6,
    "IMPORTS": 0.4,
    "DEFINES": 0.5,
}

MIN_WEIGHT = 0.05


def callers(store: GraphStore, symbol: str, depth: int = 1) -> list[str]:
    """Вызывающие функции (incoming CALLS)."""
    return store.callers(symbol, depth=depth)


def callees(store: GraphStore, symbol: str, depth: int = 1) -> list[str]:
    """Вызываемые функции (outgoing CALLS)."""
    return store.callees(symbol, depth=depth)


def subclasses(store: GraphStore, symbol: str) -> list[str]:
    """Классы, наследующие symbol (incoming INHERITS)."""
    return store.subclasses(symbol)


def imports_of(store: GraphStore, path: str) -> list[str]:
    """Файлы, которые импортирует данный путь (outgoing IMPORTS)."""
    return store.imports_of(path)


def docs_for(store: GraphStore, symbol: str) -> list[str]:
    """Документация, ссылающаяся на symbol (incoming DOCUMENTS)."""
    return store.docs_for(symbol)


@dataclass
class ExpandedHit:
    node_id: str
    score: float
    via: str


def expand(
    store: GraphStore,
    seed_ids: list[str],
    seed_scores: dict[str, float],
    hops: int = 2,
) -> list[ExpandedHit]:
    """Обход от затравочных узлов с затуханием веса.

    Возвращает список ExpandedHit, отсортированный по убыванию score.
    Причина (via) имеет вид "graph:calls", "graph:inherits" и т.п.
    """
    best: dict[str, tuple[float, str]] = {
        sid: (seed_scores.get(sid, 1.0), "seed") for sid in seed_ids
    }

    frontier: list[tuple[str, float]] = [
        (sid, seed_scores.get(sid, 1.0)) for sid in seed_ids
    ]

    for _ in range(max(hops, 1)):
        next_frontier: list[tuple[str, float]] = []
        for node_id, parent_weight in frontier:
            nbrs = store.neighbors(node_id)
            degree = len(nbrs) if nbrs else 1
            for nbr_id, edge_type, _w in nbrs:
                base = EDGE_WEIGHTS.get(edge_type)
                if base is None:
                    continue
                new_weight = parent_weight * base / (1.0 + 0.15 * (degree - 1))
                if new_weight < MIN_WEIGHT:
                    continue
                via = f"graph:{edge_type.lower()}"
                old = best.get(nbr_id)
                if old is None or new_weight > old[0]:
                    best[nbr_id] = (new_weight, via)
                    next_frontier.append((nbr_id, new_weight))
        frontier = next_frontier
        if not frontier:
            break

    out = [
        ExpandedHit(node_id=nid, score=score, via=via)
        for nid, (score, via) in best.items()
    ]
    out.sort(key=lambda h: h.score, reverse=True)
    return out