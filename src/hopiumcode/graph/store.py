"""
Хранилища графа кода: абстрактный интерфейс и две реализации.

- NetworkXStore: запасная, в памяти + сохранение в pickle. Работает без
  дополнительных зависимостей (кроме networkx).
- KuzuStore: встраиваемая графовая БД Kuzu. Если пакет kuzu не установлен,
  класс создаётся, но его конструктор бросает ImportError с понятным текстом.

Все пути и идентификаторы узлов совпадают с Node.id из extract.py:
"path" для File/Doc, "path::qualified_name" для Class/Function.
"""
from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from codeguide.graph.extract import Edge, Node

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None  # type: ignore[assignment]

try:
    import kuzu  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    kuzu = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Абстрактный интерфейс
# ---------------------------------------------------------------------------

class GraphStore(ABC):
    """Общий контракт хранилища графа (см. раздел 5 ТЗ)."""

    @abstractmethod
    def add_nodes(self, nodes: Iterable[Node]) -> None: ...

    @abstractmethod
    def add_edges(self, edges: Iterable[Edge]) -> None: ...

    @abstractmethod
    def save(self) -> None: ...

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def callers(self, symbol: str, depth: int = 1) -> list[str]: ...

    @abstractmethod
    def callees(self, symbol: str, depth: int = 1) -> list[str]: ...

    @abstractmethod
    def subclasses(self, symbol: str) -> list[str]: ...

    @abstractmethod
    def imports_of(self, path: str) -> list[str]: ...

    @abstractmethod
    def docs_for(self, symbol: str) -> list[str]: ...

    @abstractmethod
    def neighbors(self, node_id: str) -> list[tuple[str, str, float]]: ...


# ---------------------------------------------------------------------------
# Общие утилиты разрешения имён
# ---------------------------------------------------------------------------

def _resolve_symbols(known_ids: set[str], names_by_id: dict[str, str], symbol: str) -> list[str]:
    """Возвращает все id, чьё полное имя или простое имя совпадает с symbol.

    Если symbol содержит "::" или "/" — считаем его полным id.
    Иначе ищем по простому имени (часть после последнего ".").
    """
    if "::" in symbol or "/" in symbol:
        return [symbol] if symbol in known_ids else []
    matches: list[str] = []
    for nid, name in names_by_id.items():
        simple = name.split(".")[-1] if name else ""
        if nid == symbol or simple == symbol:
            matches.append(nid)
    return matches


# ---------------------------------------------------------------------------
# NetworkXStore
# ---------------------------------------------------------------------------

class NetworkXStore(GraphStore):
    """Запасная реализация: networkx.DiGraph + pickle.

    Ограничения: все данные в памяти; для графа на десятки тысяч узлов
    pickle-файл может занимать десятки МБ. Для проекта до 100 файлов
    это приемлемо.
    """

    def __init__(self, path: Path | str = "data/graph/graph.pkl"):
        if nx is None:
            raise ImportError("networkx не установлен: pip install networkx")
        self._path = Path(path)
        self._g = nx.DiGraph()
        self._names: dict[str, str] = {}

    # --- запись ------------------------------------------------------------
    def add_nodes(self, nodes: Iterable[Node]) -> None:
        for n in nodes:
            self._g.add_node(
                n.id,
                type=n.type,
                name=n.name,
                path=n.path,
                start_line=n.start_line,
                end_line=n.end_line,
            )
            self._names[n.id] = n.name

    def add_edges(self, edges: Iterable[Edge]) -> None:
        for e in edges:
            if e.src not in self._g or e.dst not in self._g:
                # узел не был добавлен — не создаём «фантомные» рёбра
                continue
            self._g.add_edge(e.src, e.dst, type=e.type, weight=1.0)

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("wb") as f:
            pickle.dump({"graph": self._g, "names": self._names}, f)

    def load(self) -> None:
        if not self._path.exists():
            raise FileNotFoundError(f"Файл графа не найден: {self._path}")
        with self._path.open("rb") as f:
            data = pickle.load(f)
        self._g = data["graph"]
        self._names = data["names"]

    # --- запросы -----------------------------------------------------------
    def _resolve(self, symbol: str) -> list[str]:
        return _resolve_symbols(set(self._g.nodes), self._names, symbol)

    def callers(self, symbol: str, depth: int = 1) -> list[str]:
        result: set[str] = set()
        frontier = set(self._resolve(symbol))
        for _ in range(max(depth, 1)):
            nxt: set[str] = set()
            for nid in frontier:
                for pred in self._g.predecessors(nid):
                    data = self._g.get_edge_data(pred, nid) or {}
                    if data.get("type") == "CALLS":
                        nxt.add(pred)
            result |= nxt
            frontier = nxt
            if not frontier:
                break
        return sorted(result)

    def callees(self, symbol: str, depth: int = 1) -> list[str]:
        result: set[str] = set()
        frontier = set(self._resolve(symbol))
        for _ in range(max(depth, 1)):
            nxt: set[str] = set()
            for nid in frontier:
                for succ in self._g.successors(nid):
                    data = self._g.get_edge_data(nid, succ) or {}
                    if data.get("type") == "CALLS":
                        nxt.add(succ)
            result |= nxt
            frontier = nxt
            if not frontier:
                break
        return sorted(result)

    def subclasses(self, symbol: str) -> list[str]:
        result: set[str] = set()
        for nid in self._resolve(symbol):
            for pred in self._g.predecessors(nid):
                data = self._g.get_edge_data(pred, nid) or {}
                if data.get("type") == "INHERITS":
                    result.add(pred)
        return sorted(result)

    def imports_of(self, path: str) -> list[str]:
        result: set[str] = set()
        for nid in self._g.nodes:
            if nid == path:
                for succ in self._g.successors(nid):
                    data = self._g.get_edge_data(nid, succ) or {}
                    if data.get("type") == "IMPORTS":
                        result.add(succ)
        return sorted(result)

    def docs_for(self, symbol: str) -> list[str]:
        result: set[str] = set()
        for nid in self._resolve(symbol):
            for pred in self._g.predecessors(nid):
                data = self._g.get_edge_data(pred, nid) or {}
                if data.get("type") == "DOCUMENTS":
                    result.add(pred)
        return sorted(result)

    def neighbors(self, node_id: str) -> list[tuple[str, str, float]]:
        if node_id not in self._g:
            return []
        out: list[tuple[str, str, float]] = []
        for succ in self._g.successors(node_id):
            data = self._g.get_edge_data(node_id, succ) or {}
            out.append((succ, data.get("type", ""), data.get("weight", 1.0)))
        for pred in self._g.predecessors(node_id):
            data = self._g.get_edge_data(pred, node_id) or {}
            out.append((pred, data.get("type", ""), data.get("weight", 1.0)))
        return out


# ---------------------------------------------------------------------------
# KuzuStore
# ---------------------------------------------------------------------------

class KuzuStore(GraphStore):
    """Встраиваемая графовая БД Kuzu.

    Схема создаётся при первом обращении:
      NODE TABLE CodeNode(id STRING, type STRING, name STRING, path STRING,
                          start_line INT64, end_line INT64, PRIMARY KEY(id))
      REL TABLE CodeEdge(FROM CodeNode TO CodeNode, type STRING, weight DOUBLE)

    Ограничения: Kuzu не поддерживает динамические имена rel-таблиц в
    параметрах, поэтому тип ребра хранится как свойство, а не как отдельная
    таблица. Для запросов callers/callees используется фильтр по свойству.
    """

    NODE_TABLE = "CodeNode"
    REL_TABLE = "CodeEdge"

    def __init__(self, db_path: Path | str = "data/graph/kuzu"):
        if kuzu is None:
            raise ImportError(
                "Kuzu не установлен. Установите: pip install kuzu, "
                "или используйте NetworkXStore."
            )
        self._db_path = Path(db_path)
        self._db_path.mkdir(parents=True, exist_ok=True)
        self._db = kuzu.Database(str(self._db_path))
        self._conn = kuzu.Connection(self._db)
        self._schema_ready = False
        self._names: dict[str, str] = {}

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        self._conn.execute(
            f"CREATE NODE TABLE IF NOT EXISTS {self.NODE_TABLE}("
            "id STRING, type STRING, name STRING, path STRING, "
            "start_line INT64, end_line INT64, PRIMARY KEY(id))"
        )
        self._conn.execute(
            f"CREATE REL TABLE IF NOT EXISTS {self.REL_TABLE}("
            f"FROM {self.NODE_TABLE} TO {self.NODE_TABLE}, "
            "type STRING, weight DOUBLE)"
        )
        self._schema_ready = True

    # --- запись ------------------------------------------------------------
    def add_nodes(self, nodes: Iterable[Node]) -> None:
        self._ensure_schema()
        for n in nodes:
            self._conn.execute(
                f"MERGE (n:{self.NODE_TABLE} {{id: $id}}) "
                "SET n.type=$type, n.name=$name, n.path=$path, "
                "n.start_line=$s, n.end_line=$e",
                {
                    "id": n.id, "type": n.type, "name": n.name, "path": n.path,
                    "s": n.start_line, "e": n.end_line,
                },
            )
            self._names[n.id] = n.name

    def add_edges(self, edges: Iterable[Edge]) -> None:
        self._ensure_schema()
        for e in edges:
            self._conn.execute(
                f"MATCH (a:{self.NODE_TABLE} {{id:$src}}), "
                f"(b:{self.NODE_TABLE} {{id:$dst}}) "
                f"MERGE (a)-[r:{self.REL_TABLE} {{type:$t}}]->(b) "
                "SET r.weight=1.0",
                {"src": e.src, "dst": e.dst, "t": e.type},
            )

    def save(self) -> None:
        # Kuzu сохраняет данные на диск автоматически при commit транзакции.
        self._conn.execute("CHECKPOINT")

    def load(self) -> None:
        # Данные уже на диске; достаточно убедиться, что схема есть,
        # и восстановить кэш имён для resolve.
        self._ensure_schema()
        result = self._conn.execute(
            f"MATCH (n:{self.NODE_TABLE}) RETURN n.id, n.name"
        )
        self._names.clear()
        while result.has_next():
            nid, name = result.get_next()
            self._names[nid] = name

    # --- запросы -----------------------------------------------------------
    def _resolve(self, symbol: str) -> list[str]:
        if "::" in symbol or "/" in symbol:
            res = self._conn.execute(
                f"MATCH (n:{self.NODE_TABLE} {{id:$id}}) RETURN n.id",
                {"id": symbol},
            )
            out = []
            while res.has_next():
                out.append(res.get_next()[0])
            return out
        # по простому имени: ищем name, оканчивающееся на .symbol или равное symbol
        res = self._conn.execute(
            f"MATCH (n:{self.NODE_TABLE}) "
            "WHERE n.name = $s OR n.name ENDS WITH $suffix "
            "RETURN n.id",
            {"s": symbol, "suffix": "." + symbol},
        )
        out = []
        while res.has_next():
            out.append(res.get_next()[0])
        return out

    def callers(self, symbol: str, depth: int = 1) -> list[str]:
        seeds = self._resolve(symbol)
        if not seeds:
            return []
        # 1 шаг: все входящие CALLS; для depth > 1 повторяем BFS в Python
        result: set[str] = set()
        frontier = set(seeds)
        for _ in range(max(depth, 1)):
            nxt: set[str] = set()
            for nid in frontier:
                res = self._conn.execute(
                    f"MATCH (a:{self.NODE_TABLE})-"
                    f"[r:{self.REL_TABLE} {{type:'CALLS'}}]->"
                    f"(b:{self.NODE_TABLE} {{id:$id}}) RETURN a.id",
                    {"id": nid},
                )
                while res.has_next():
                    nxt.add(res.get_next()[0])
            result |= nxt
            frontier = nxt
            if not frontier:
                break
        return sorted(result)

    def callees(self, symbol: str, depth: int = 1) -> list[str]:
        seeds = self._resolve(symbol)
        if not seeds:
            return []
        result: set[str] = set()
        frontier = set(seeds)
        for _ in range(max(depth, 1)):
            nxt: set[str] = set()
            for nid in frontier:
                res = self._conn.execute(
                    f"MATCH (a:{self.NODE_TABLE} {{id:$id}})-"
                    f"[r:{self.REL_TABLE} {{type:'CALLS'}}]->"
                    f"(b:{self.NODE_TABLE}) RETURN b.id",
                    {"id": nid},
                )
                while res.has_next():
                    nxt.add(res.get_next()[0])
            result |= nxt
            frontier = nxt
            if not frontier:
                break
        return sorted(result)

    def subclasses(self, symbol: str) -> list[str]:
        seeds = self._resolve(symbol)
        if not seeds:
            return []
        result: set[str] = set()
        for nid in seeds:
            res = self._conn.execute(
                f"MATCH (a:{self.NODE_TABLE})-"
                f"[r:{self.REL_TABLE} {{type:'INHERITS'}}]->"
                f"(b:{self.NODE_TABLE} {{id:$id}}) RETURN a.id",
                {"id": nid},
            )
            while res.has_next():
                result.add(res.get_next()[0])
        return sorted(result)

    def imports_of(self, path: str) -> list[str]:
        res = self._conn.execute(
            f"MATCH (a:{self.NODE_TABLE} {{id:$id}})-"
            f"[r:{self.REL_TABLE} {{type:'IMPORTS'}}]->"
            f"(b:{self.NODE_TABLE}) RETURN b.id",
            {"id": path},
        )
        out: list[str] = []
        while res.has_next():
            out.append(res.get_next()[0])
        return sorted(out)

    def docs_for(self, symbol: str) -> list[str]:
        seeds = self._resolve(symbol)
        if not seeds:
            return []
        result: set[str] = set()
        for nid in seeds:
            res = self._conn.execute(
                f"MATCH (a:{self.NODE_TABLE})-"
                f"[r:{self.REL_TABLE} {{type:'DOCUMENTS'}}]->"
                f"(b:{self.NODE_TABLE} {{id:$id}}) RETURN a.id",
                {"id": nid},
            )
            while res.has_next():
                result.add(res.get_next()[0])
        return sorted(result)

    def neighbors(self, node_id: str) -> list[tuple[str, str, float]]:
        out: list[tuple[str, str, float]] = []
        res = self._conn.execute(
            f"MATCH (a:{self.NODE_TABLE} {{id:$id}})-"
            f"[r:{self.REL_TABLE}]->(b:{self.NODE_TABLE}) "
            "RETURN b.id, r.type, r.weight",
            {"id": node_id},
        )
        while res.has_next():
            out.append(tuple(res.get_next()))  # type: ignore[arg-type]
        res = self._conn.execute(
            f"MATCH (a:{self.NODE_TABLE})-"
            f"[r:{self.REL_TABLE}]->(b:{self.NODE_TABLE} {{id:$id}}) "
            "RETURN a.id, r.type, r.weight",
            {"id": node_id},
        )
        while res.has_next():
            out.append(tuple(res.get_next()))  # type: ignore[arg-type]
        return out