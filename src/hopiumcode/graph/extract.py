"""
Извлечение узлов и рёбер графа кода без использования LLM.

Ограничения (важно):
- Анализируется только Python-код (модуль ast). Markdown используется только
  для рёбер DOCUMENTS.
- Идентификаторы узлов совпадают с Chunk.id из chunker: "path::qualified_name"
  для символов и "path::#N" не используются здесь — для файла id = "path".
- CALLS разрешаются только в пределах видимости: локальные функции модуля,
  импортированные имена (from x import y), self.method внутри класса.
  Неразрешённые вызовы НЕ угадываются и пропускаются.
"""
from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

logger = logging.getLogger(__name__)

NodeType = Literal["File", "Class", "Function", "Doc"]
EdgeType = Literal["DEFINES", "IMPORTS", "INHERITS", "CALLS", "DOCUMENTS"]


@dataclass(frozen=True)
class Node:
    id: str
    type: NodeType
    name: str
    path: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    type: EdgeType


# ---------------------------------------------------------------------------
# Вспомогательные структуры
# ---------------------------------------------------------------------------

@dataclass
class _FileInfo:
    """Собранная информация по одному Python-файлу."""
    path: str                       # относительный путь
    module_dotted: str              # "pkg.sub.module"
    node: Node
    defines: list[Edge] = field(default_factory=list)
    imports: list[Edge] = field(default_factory=list)
    # локальные символы: simple_name -> full_id
    local_symbols: dict[str, str] = field(default_factory=dict)
    # импортированные имена: simple_name -> (source_module_dotted, original_name)
    imported_symbols: dict[str, tuple[str, str]] = field(default_factory=dict)
    # базовые классы классов: class_id -> [base_simple_name]
    class_bases: dict[str, list[str]] = field(default_factory=dict)
    # вызовы: (caller_id, callee_simple_name, is_self_method)
    raw_calls: list[tuple[str, str, bool]] = field(default_factory=list)


def _node_id(path: str, qualified_name: str | None = None) -> str:
    return path if qualified_name is None else f"{path}::{qualified_name}"


def _module_dotted(rel_path: str) -> str:
    p = rel_path[:-3] if rel_path.endswith(".py") else rel_path
    return p.replace("/", ".").replace("\\", ".")


# ---------------------------------------------------------------------------
# Парсинг одного Python-файла
# ---------------------------------------------------------------------------

class _PythonVisitor(ast.NodeVisitor):
    def __init__(self, info: _FileInfo):
        self.info = info
        self.class_stack: list[str] = []
        self.func_stack: list[str] = []

    # --- определения -------------------------------------------------------
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qual = ".".join(self.class_stack + [node.name])
        cls_id = _node_id(self.info.path, qual)
        self.info.local_symbols[node.name] = cls_id
        self.info.defines.append(
            Edge(src=self.info.node.id, dst=cls_id, type="DEFINES")
        )
        # родительский контейнер (модуль или внешний класс)
        parent_qual = ".".join(self.class_stack)
        parent_id = (
            _node_id(self.info.path, parent_qual)
            if parent_qual
            else self.info.node.id
        )
        self.info.defines.append(Edge(src=parent_id, dst=cls_id, type="DEFINES"))

        self.info.class_bases[cls_id] = [
            _base_name(b) for b in node.bases if _base_name(b)
        ]

        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._handle_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._handle_function(node)

    def _handle_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qual = ".".join(self.class_stack + self.func_stack + [node.name])
        func_id = _node_id(self.info.path, qual)
        self.info.local_symbols.setdefault(node.name, func_id)

        parent_qual = ".".join(self.class_stack + self.func_stack)
        parent_id = (
            _node_id(self.info.path, parent_qual)
            if parent_qual
            else self.info.node.id
        )
        self.info.defines.append(Edge(src=parent_id, dst=func_id, type="DEFINES"))

        self.func_stack.append(node.name)
        self.generic_visit(node)
        self.func_stack.pop()

    # --- импорты -----------------------------------------------------------
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            simple = alias.asname or alias.name.split(".")[0]
            self.info.imported_symbols[simple] = (alias.name, alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is None and node.level == 0:
            return
        base = node.module or ""
        if node.level:
            # относительный импорт: восстановим пакет текущего файла
            pkg_parts = self.info.module_dotted.split(".")[:-1]
            drop = node.level - 1
            if drop > len(pkg_parts):
                return
            base_pkg = pkg_parts[: len(pkg_parts) - drop] if drop else pkg_parts
            full = ".".join(base_pkg + ([base] if base else []))
        else:
            full = base
        for alias in node.names:
            simple = alias.asname or alias.name
            self.info.imported_symbols[simple] = (full, alias.name)
        self.generic_visit(node)

    # --- вызовы ------------------------------------------------------------
    def visit_Call(self, node: ast.Call) -> None:
        if self.func_stack:
            caller_qual = ".".join(self.class_stack + self.func_stack)
            caller_id = _node_id(self.info.path, caller_qual)
            name, is_self = _call_target_name(node.func)
            if name:
                self.info.raw_calls.append((caller_id, name, is_self))
        self.generic_visit(node)


def _base_name(expr: ast.expr) -> str | None:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        return expr.attr
    return None


def _call_target_name(func: ast.expr) -> tuple[str | None, bool]:
    """Возвращает (simple_name, is_self_method)."""
    if isinstance(func, ast.Name):
        return func.id, False
    if isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name) and func.value.id == "self":
            return func.attr, True
        return None, False
    return None, False


# ---------------------------------------------------------------------------
# Основная функция
# ---------------------------------------------------------------------------

def extract(root: Path) -> tuple[list[Node], list[Edge]]:
    root = root.resolve()
    py_files = sorted(
        p for p in root.rglob("*.py")
        if not any(part in {".git", "__pycache__", ".venv", "node_modules"} for part in p.parts)
    )
    md_files = sorted(
        p for p in root.rglob("*.md")
        if not any(part in {".git", "__pycache__", ".venv", "node_modules"} for part in p.parts)
    )

    infos: list[_FileInfo] = []
    for p in py_files:
        rel = p.relative_to(root).as_posix()
        try:
            source = p.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            tree = ast.parse(source, filename=rel)
        except SyntaxError as exc:
            logger.warning("Пропущен файл с синтаксической ошибкой: %s (%s)", rel, exc)
            continue

        n_lines = source.count("\n") + 1
        fnode = Node(
            id=_node_id(rel),
            type="File",
            name=p.name,
            path=rel,
            start_line=1,
            end_line=n_lines,
        )
        info = _FileInfo(path=rel, module_dotted=_module_dotted(rel), node=fnode)
        _PythonVisitor(info).visit(tree)
        infos.append(info)

    nodes: list[Node] = []
    edges: list[Edge] = []

    # --- узлы и DEFINES ----------------------------------------------------
    for info in infos:
        nodes.append(info.node)
        edges.extend(info.defines)
        for cls_id in info.class_bases:
            qual = cls_id.split("::", 1)[1]
            short = qual.split(".")[-1]
            nodes.append(Node(
                id=cls_id, type="Class", name=short, path=info.path,
                start_line=0, end_line=0,
            ))
        # Function-узлы
        for e in info.defines:
            if e.type != "DEFINES":
                continue
            dst = e.dst
            if dst == info.node.id:
                continue
            if "::" not in dst:
                continue
            qual = dst.split("::", 1)[1]
            short = qual.split(".")[-1]
            is_method = "." in qual
            nodes.append(Node(
                id=dst,
                type="Function",
                name=short,
                path=info.path,
                start_line=0,
                end_line=0,
            ))

    # дедупликация узлов по id
    seen: set[str] = set()
    unique_nodes: list[Node] = []
    for n in nodes:
        if n.id in seen:
            continue
        seen.add(n.id)
        unique_nodes.append(n)
    nodes = unique_nodes

    # --- IMPORTS -----------------------------------------------------------
    by_module: dict[str, _FileInfo] = {i.module_dotted: i for i in infos}
    for info in infos:
        for simple, (src_mod, _orig) in info.imported_symbols.items():
            target = by_module.get(src_mod)
            if target is not None and target.path != info.path:
                info.imports.append(
                    Edge(src=info.node.id, dst=target.node.id, type="IMPORTS")
                )
    for info in infos:
        edges.extend(info.imports)

    # --- INHERITS ----------------------------------------------------------
    global_symbols: dict[str, str] = {}
    for info in infos:
        for simple, full_id in info.local_symbols.items():
            global_symbols.setdefault(simple, full_id)

    for info in infos:
        for cls_id, bases in info.class_bases.items():
            for base in bases:
                target: str | None = None
                if base in info.imported_symbols:
                    src_mod, orig = info.imported_symbols[base]
                    owner = by_module.get(src_mod)
                    if owner and orig in owner.local_symbols:
                        target = owner.local_symbols[orig]
                if target is None:
                    target = info.local_symbols.get(base)
                if target is None:
                    target = global_symbols.get(base)
                if target and target != cls_id:
                    edges.append(Edge(src=cls_id, dst=target, type="INHERITS"))

    # --- CALLS -------------------------------------------------------------
    # пересоберём таблицу: simple_name -> full_id для функций
    func_index: dict[str, str] = {}
    for info in infos:
        for e in info.defines:
            if e.type != "DEFINES":
                continue
            dst = e.dst
            if "::" not in dst:
                continue
            qual = dst.split("::", 1)[1]
            short = qual.split(".")[-1]
            func_index.setdefault(short, dst)

    for info in infos:
        for caller_id, callee_name, is_self in info.raw_calls:
            target: str | None = None
            if is_self:
                # self.method -> ищем метод в текущем классе
                if "::" in caller_id:
                    qual = caller_id.split("::", 1)[1]
                    parts = qual.split(".")
                    if len(parts) >= 2:
                        cls_qual = ".".join(parts[:-1])
                        candidate_qual = f"{cls_qual}.{callee_name}"
                        cand_id = _node_id(info.path, candidate_qual)
                        for e in info.defines:
                            if e.dst == cand_id:
                                target = cand_id
                                break
            if target is None and callee_name in info.local_symbols:
                target = info.local_symbols[callee_name]
            if target is None and callee_name in info.imported_symbols:
                src_mod, orig = info.imported_symbols[callee_name]
                owner = by_module.get(src_mod)
                if owner and orig in owner.local_symbols:
                    target = owner.local_symbols[orig]
            if target is None:
                # глобальный поиск по имени — только если имя уникально
                matches = [fid for nm, fid in func_index.items() if nm == callee_name]
                if len(matches) == 1:
                    target = matches[0]
            if target is None or target == caller_id:
                continue
            edges.append(Edge(src=caller_id, dst=target, type="CALLS"))

    # --- DOCUMENTS ---------------------------------------------------------
    symbol_names = {n.name for n in nodes if n.type in {"Class", "Function"} and n.name}
    for p in md_files:
        rel = p.relative_to(root).as_posix()
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        doc_node = Node(
            id=_node_id(rel),
            type="Doc",
            name=p.name,
            path=rel,
            start_line=1,
            end_line=text.count("\n") + 1,
        )
        nodes.append(doc_node)
        for name in sorted(symbol_names):
            marker = f"`{name}`"
            if marker in text:
                for n in nodes:
                    if n.name == name and n.type in {"Class", "Function"}:
                        edges.append(Edge(src=doc_node.id, dst=n.id, type="DOCUMENTS"))

    # дедупликация рёбер
    seen_e: set[tuple[str, str, str]] = set()
    unique_edges: list[Edge] = []
    for e in edges:
        key = (e.src, e.dst, e.type)
        if key in seen_e:
            continue
        seen_e.add(key)
        unique_edges.append(e)

    return nodes, unique_edges