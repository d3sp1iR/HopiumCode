"""Хранилище графа за общим интерфейсом"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx

from ..config import Config
from ..models import Edge, Node


@dataclass(frozen=True)
class Resolution:
    ids: list[str]
    ambiguous: bool


class GraphStore(ABC):
    @abstractmethod
    def add_nodes(self, nodes: Iterable[Node]) -> None: ...

    @abstractmethod
    def add_edges(self, edges: Iterable[Edge]) -> None: ...

    @abstractmethod
    def save(self) -> None: ...

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def get_node(self, node_id: str) -> Node | None: ...

    @abstractmethod
    def find_by_name(self, name: str) -> list[str]: ...

    @abstractmethod
    def degree(self, node_id: str) -> int: ...

    @abstractmethod
    def neighbors(self, node_id: str) -> list[tuple[str, str]]: ...

    @abstractmethod
    def related(self, node_id: str, etype: str, incoming: bool) -> list[str]: ...

    @abstractmethod
    def iter_edges(self, etype: str | None = None) -> list[Edge]: ...

    def resolve(self, symbol: str) -> Resolution:
        if self.get_node(symbol):
            return Resolution([symbol], False)
        ids = self.find_by_name(symbol)
        return Resolution(ids, len(ids) > 1)

    def _walk(self, symbol: str, etype: str, incoming: bool, depth: int) -> list[str]:
        frontier = self.resolve(symbol).ids
        seen = set(frontier)
        order: list[str] = []
        for _ in range(depth):
            nxt: list[str] = []
            for node in frontier:
                for other in self.related(node, etype, incoming):
                    if other not in seen:
                        seen.add(other)
                        order.append(other)
                        nxt.append(other)
            frontier = nxt
        return order

    def callers(self, symbol: str, depth: int = 1) -> list[str]:
        return self._walk(symbol, "CALLS", True, depth)

    def callees(self, symbol: str, depth: int = 1) -> list[str]:
        return self._walk(symbol, "CALLS", False, depth)

    def subclasses(self, symbol: str, depth: int = 1) -> list[str]:
        return self._walk(symbol, "INHERITS", True, depth)

    def imports_of(self, path: str) -> list[str]:
        return self._walk(path, "IMPORTS", False, 1)

    def docs_for(self, symbol: str) -> list[str]:
        return self._walk(symbol, "DOCUMENTS", True, 1)

    def expand(
        self, seed_ids: list[str], seed_scores: dict[str, float], hops: int = 2
    ) -> list[tuple[str, float, str]]:
        from .query import expand

        return expand(seed_ids, seed_scores, hops, store=self)


class NetworkXStore(GraphStore):
    """Граф в памяти (networkx) с сохранением в JSON."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_graph_path()
        self.g: nx.MultiDiGraph = nx.MultiDiGraph()
        self._names: dict[str, list[str]] | None = None

    def add_nodes(self, nodes: Iterable[Node]) -> None:
        for node in nodes:
            attrs = asdict(node)
            attrs.pop("id")
            self.g.add_node(node.id, **attrs)
        self._names = None

    def add_edges(self, edges: Iterable[Edge]) -> None:
        for e in edges:
            if e.src in self.g and e.dst in self.g:
                self.g.add_edge(e.src, e.dst, key=e.type)

    def get_node(self, node_id: str) -> Node | None:
        attrs = self.g.nodes.get(node_id)
        return Node(node_id, **attrs) if attrs is not None else None

    def find_by_name(self, name: str) -> list[str]:
        if self._names is None:
            index: dict[str, list[str]] = {}
            for nid, attrs in self.g.nodes(data=True):
                if attrs["type"] in ("Class", "Function"):
                    parts = attrs["name"].split(".")
                    for i in range(len(parts)):
                        index.setdefault(".".join(parts[i:]), []).append(nid)
            self._names = index
        return list(self._names.get(name, []))

    def degree(self, node_id: str) -> int:
        return self.g.degree(node_id) if node_id in self.g else 0

    def neighbors(self, node_id: str) -> list[tuple[str, str]]:
        if node_id not in self.g:
            return []
        out = [(v, k) for _, v, k in self.g.out_edges(node_id, keys=True)]
        inc = [(u, k) for u, _, k in self.g.in_edges(node_id, keys=True)]
        return out + inc

    def related(self, node_id: str, etype: str, incoming: bool) -> list[str]:
        if node_id not in self.g:
            return []
        if incoming:
            return [u for u, _, k in self.g.in_edges(node_id, keys=True) if k == etype]
        return [v for _, v, k in self.g.out_edges(node_id, keys=True) if k == etype]

    def iter_edges(self, etype: str | None = None) -> list[Edge]:
        return [Edge(u, v, k) for u, v, k in self.g.edges(keys=True) if etype in (None, k)]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        nodes = [asdict(self.get_node(n)) for n in self.g.nodes]
        edges = [asdict(e) for e in self.iter_edges()]
        payload = json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False)
        self.path.write_text(payload, encoding="utf-8")

    def load(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"граф не построен: {self.path} (выполните make index)")
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.g.clear()
        self.add_nodes(Node(**n) for n in data["nodes"])
        self.add_edges(Edge(**e) for e in data["edges"])


class KuzuStore(NetworkXStore):
    """Kuzu заархивирован в октябре 2025, поэтому не реализуем: см. docs/decisions.md."""

    def __init__(self, path: Path | None = None) -> None:
        raise NotImplementedError("KuzuStore не реализован: используйте CG_GRAPH_BACKEND=networkx")


def default_graph_path(config: Config | None = None) -> Path:
    return (config or Config.from_env()).graph_dir / "graph.json"


def open_store(config: Config | None = None) -> GraphStore:
    cfg = config or Config.from_env()
    path = default_graph_path(cfg)
    if cfg.graph_backend == "networkx":
        return NetworkXStore(path)
    if cfg.graph_backend == "kuzu":
        return KuzuStore(path)
    raise ValueError(f"неизвестный graph_backend: {cfg.graph_backend}")