"""Общие данные. Владелец: участник 1. Менять только по согласованию всех зон."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ChunkKind = Literal["function", "method", "class", "module", "doc"]
NodeType = Literal["File", "Class", "Function", "Doc"]
EdgeType = Literal["DEFINES", "CALLS", "IMPORTS", "INHERITS", "DOCUMENTS"]


@dataclass(frozen=True)
class Chunk:
    id: str            # "path::qualified_name" или "path::#N"
    path: str          # относительно корня репозитория-цели
    start_line: int
    end_line: int
    kind: ChunkKind
    name: str | None   # квалифицированное имя символа
    text: str
    lang: str          # "python" | "markdown" | ...

    @property
    def ref(self) -> str:
        """Ссылка для ответа: [путь:начало-конец]."""
        return f"[{self.path}:{self.start_line}-{self.end_line}]"


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float       # 0..1, больше = лучше
    via: str           # "vector" | "bm25" | "graph:calls" | ...


@dataclass(frozen=True)
class Node:
    id: str            # совпадает с Chunk.id для символов
    type: NodeType
    name: str
    path: str
    start_line: int = 0
    end_line: int = 0


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    type: EdgeType


@dataclass
class Answer:
    text: str
    citations: list[str] = field(default_factory=list)   # допустимые ссылки
    bad_citations: list[str] = field(default_factory=list)
    used_tools: list[str] = field(default_factory=list)
    steps: int = 0
    latency_s: float = 0.0
