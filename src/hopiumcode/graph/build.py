"""Сборка графа связей"""
from __future__ import annotations

from collections import Counter

from ..config import Config
from ..indexing.extract import extract
from ..indexing.store import open_store


def build_graph(config: Config | None = None) -> dict[str, dict[str, int]]:
    cfg = config or Config.from_env()
    nodes, edges = extract(cfg.repo_root)
    store = open_store(cfg)
    store.add_nodes(nodes)
    store.add_edges(edges)
    store.save()
    return {
        "nodes": dict(Counter(n.type for n in nodes)),
        "edges": dict(Counter(e.type for e in edges)),
    }