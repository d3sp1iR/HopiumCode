"""Извлечение символов и связей из кода"""
from __future__ import annotations

import ast
import itertools
import logging
import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from ..indexing.loader import MAX_BYTES, SKIP_DIRS
from ..models import Edge, Node

log = logging.getLogger(__name__)

_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_MD_HEADING = re.compile(r"#{1,6}\s")
_RST_ADORN = re.compile(r"([=\-`:'\"~^_*+#<>])\1{2,}\s*")
_FENCE = re.compile(r"```.*?```", re.DOTALL)
_SPAN = re.compile(r"`([^`\n]+)`")
_DIRECTIVE = re.compile(r"^\s*\.\. auto\w+::\s*(\S+)", re.MULTILINE)
_REF = re.compile(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*")

Target = tuple[str, str]
Section = tuple[int, int, str, str]


@dataclass(frozen=True)
class _Import:
    local: str
    module: str
    name: str | None


def _iter_files(root: Path, suffix: str) -> Iterator[tuple[Path, str]]:
    for path in sorted(root.rglob(f"*{suffix}")):
        rel = path.relative_to(root)
        if not path.is_file() or any(part in SKIP_DIRS for part in rel.parts[:-1]):
            continue
        if path.stat().st_size > MAX_BYTES:
            continue
        yield path, rel.as_posix()


def _blocks(stmt: ast.stmt) -> Iterator[list[ast.stmt]]:
    for field in ("body", "orelse", "finalbody"):
        block = getattr(stmt, field, None)
        if block:
            yield block
    for handler in getattr(stmt, "handlers", []):
        yield handler.body


def _calls(root: ast.AST, into_defs: bool) -> list[ast.Call]:
    found: list[ast.Call] = []
    stack = list(ast.iter_child_nodes(root))
    while stack:
        node = stack.pop()
        if not into_defs and isinstance(node, _DEFS):
            continue
        if isinstance(node, ast.Call):
            found.append(node)
        stack.extend(ast.iter_child_nodes(node))
    return sorted(found, key=lambda c: (c.lineno, c.col_offset))


def _dotted(expr: ast.expr) -> str | None:
    parts: list[str] = []
    while isinstance(expr, ast.Attribute):
        parts.append(expr.attr)
        expr = expr.value
    if isinstance(expr, ast.Name):
        parts.append(expr.id)
        return ".".join(reversed(parts))
    return None


def _split(lines: list[str], cuts: list[tuple[int, str]]) -> list[Section]:
    bounds = [c for c, _ in cuts]
    titles = dict(cuts)
    if not bounds or bounds[0] > 0:
        bounds.insert(0, 0)
        titles.setdefault(0, "")
    bounds.append(len(lines))
    out: list[Section] = []
    for a, b in itertools.pairwise(bounds):
        block = lines[a:b]
        if any(line.strip() for line in block):
            out.append((a + 1, b, titles[a], "\n".join(block)))
    return out


def _md_sections(text: str) -> list[Section]:
    lines = text.splitlines()
    cuts: list[tuple[int, str]] = []
    fenced = False
    for i, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            fenced = not fenced
        elif not fenced and _MD_HEADING.match(line):
            cuts.append((i, line.lstrip("# ").strip()))
    return _split(lines, cuts)


def _rst_sections(text: str) -> list[Section]:
    lines = text.splitlines()
    cuts: list[tuple[int, str]] = []
    for i in range(len(lines) - 1):
        title, under = lines[i], lines[i + 1]
        is_title = title.strip() and not title[0].isspace() and not _RST_ADORN.fullmatch(title)
        if is_title and _RST_ADORN.fullmatch(under) and len(under.rstrip()) >= len(title.rstrip()):
            over = i > 0 and _RST_ADORN.fullmatch(lines[i - 1]) and lines[i - 1][0] == under[0]
            cuts.append((i - 1 if over else i, title.strip()))
    return _split(lines, cuts)


def _doc_refs(body: str, rst: bool) -> list[str]:
    refs = _SPAN.findall(_FENCE.sub("", body))
    if rst:
        refs += _DIRECTIVE.findall(body)
    out = []
    for ref in refs:
        ref = ref.strip()
        angle = re.search(r"<([^<>]+)>$", ref)
        out.append((angle.group(1) if angle else ref).lstrip("~.!@"))
    return out


class _Builder:
    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: dict[tuple[str, str, str], Edge] = {}
        self.call_lines: dict[tuple[str, str], int] = {}
        self.trees: dict[str, ast.Module] = {}
        self.dotted: dict[str, str] = {}
        self.modmap: dict[str, str | None] = {}
        self.top: dict[str, dict[str, str]] = {}
        self.members: dict[str, dict[str, str]] = {}
        self.raw: dict[str, list[_Import]] = {}
        self.env: dict[str, dict[str, Target]] = {}
        self.bases: dict[str, list[str]] = {}
        self.funcs: list[tuple[str, str, ast.AST, str | None]] = []
        self.classes: list[tuple[str, str, ast.ClassDef]] = []

    def add_edge(self, src: str, dst: str, etype: str, line: int | None = None) -> None:
        key = (src, dst, etype)
        if key not in self.edges:
            self.edges[key] = Edge(src, dst, etype)
            if line is not None:
                self.call_lines[(src, dst)] = line

    def add_module(self, path: Path, rel: str) -> None:
        try:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text, filename=rel)
        except (SyntaxError, UnicodeDecodeError, ValueError) as exc:
            log.warning("файл пропущен %s: %s", rel, exc)
            return
        self.trees[rel] = tree
        self.nodes[rel] = Node(rel, "File", rel, rel, 1, max(1, len(text.splitlines())))

    def index_modules(self) -> None:
        files = set(self.trees)
        for rel in self.trees:
            p = Path(rel)
            names: list[str] = []
            d = p.parent
            while d != Path(".") and (d / "__init__.py").as_posix() in files:
                names.insert(0, d.name)
                d = d.parent
            if p.stem != "__init__":
                names.append(p.stem)
            dotted = ".".join(names)
            self.dotted[rel] = dotted
            if dotted:
                self.modmap[dotted] = None if self.modmap.get(dotted, rel) != rel else rel

    def collect_defs(self, rel: str, tree: ast.Module) -> None:
        """DEFINES: файл -> класс/функция, класс -> метод и вложенный класс.
        Функции, вложенные в функции, отдельными узлами не становятся: они часть родителя."""
        self.top[rel] = {}

        def visit(body: list[ast.stmt], prefix: str, parent: str, cls: str | None) -> None:
            for st in body:
                if isinstance(st, _DEFS):
                    qn = prefix + st.name
                    sid = f"{rel}::{qn}"
                    is_cls = isinstance(st, ast.ClassDef)
                    start = min([st.lineno] + [d.lineno for d in st.decorator_list])
                    self.nodes[sid] = Node(
                        sid, "Class" if is_cls else "Function", qn, rel, start,
                        st.end_lineno or st.lineno,
                    )
                    self.add_edge(parent, sid, "DEFINES")
                    table = self.top[rel] if parent == rel else self.members.setdefault(parent, {})
                    table[st.name] = sid
                    if is_cls:
                        self.classes.append((rel, sid, st))
                        visit(st.body, qn + ".", sid, sid)
                    else:
                        self.funcs.append((rel, sid, st, cls))
                elif isinstance(st, (ast.If, ast.Try, ast.With)):
                    for block in _blocks(st):
                        visit(block, prefix, parent, cls)

        visit(tree.body, "", rel, None)

    def package_of(self, rel: str) -> str:
        dotted = self.dotted.get(rel, "")
        return dotted if rel.endswith("__init__.py") else dotted.rpartition(".")[0]

    @staticmethod
    def absolute(pkg: str, module: str | None, level: int) -> str | None:
        if level == 0:
            return module
        parts = pkg.split(".") if pkg else []
        if level - 1 > len(parts):
            return None
        parts = parts[: len(parts) - (level - 1)]
        if module:
            parts += module.split(".")
        return ".".join(parts)

    def collect_imports(self, rel: str, tree: ast.Module) -> None:
        pkg = self.package_of(rel)
        found: list[_Import] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        found.append(_Import(alias.asname, alias.name, None))
                        continue
                    parts = alias.name.split(".")
                    for i in range(1, len(parts) + 1):
                        prefix = ".".join(parts[:i])
                        found.append(_Import(prefix, prefix, None))
            elif isinstance(node, ast.ImportFrom):
                base = self.absolute(pkg, node.module, node.level)
                if base is None:
                    continue
                for alias in node.names:
                    if alias.name != "*":
                        found.append(_Import(alias.asname or alias.name, base, alias.name))
        self.raw[rel] = found

    def member(self, fpath: str, name: str, depth: int = 0) -> Target | None:
        if name in self.top.get(fpath, {}):
            return ("sym", self.top[fpath][name])
        if fpath.endswith("__init__.py") and self.dotted.get(fpath):
            sub = self.modmap.get(f"{self.dotted[fpath]}.{name}")
            if sub:
                return ("mod", sub)
        if depth < 5:
            for imp in self.raw.get(fpath, []):
                if imp.local == name:
                    found = self.entry(imp, depth + 1)
                    if found:
                        return found
        return None

    def entry(self, imp: _Import, depth: int) -> Target | None:
        fpath = self.modmap.get(imp.module)
        if fpath is None:
            return None
        if imp.name is None:
            return ("mod", fpath)
        return self.member(fpath, imp.name, depth)

    def resolve_imports(self) -> None:
        for rel, imports in self.raw.items():
            env: dict[str, Target] = {}
            for imp in imports:
                found = self.entry(imp, 0)
                if found:
                    env[imp.local] = found
            self.env[rel] = env

    def link_imports(self) -> None:
        """IMPORTS: файл -> файл репозитория, указанный в операторе import.
        Внешние библиотеки не учитываются; динамические импорты (importlib) не видны."""
        for rel, imports in self.raw.items():
            for imp in imports:
                fpath = self.modmap.get(imp.module)
                if fpath is None:
                    continue
                if fpath != rel:
                    self.add_edge(rel, fpath, "IMPORTS")
                if imp.name:
                    found = self.member(fpath, imp.name)
                    if found and found[0] == "mod" and found[1] != rel:
                        self.add_edge(rel, found[1], "IMPORTS")

    def target(self, name: str, rel: str) -> Target | None:
        if "." not in name and name in self.top.get(rel, {}):
            return ("sym", self.top[rel][name])
        return self.env.get(rel, {}).get(name)

    def method(
        self, cid: str, name: str, include_self: bool = True, seen: set[str] | None = None
    ) -> str | None:
        seen = seen if seen is not None else set()
        if cid in seen:
            return None
        seen.add(cid)
        if include_self and name in self.members.get(cid, {}):
            return self.members[cid][name]
        for base in self.bases.get(cid, []):
            found = self.method(base, name, True, seen)
            if found:
                return found
        return None

    def ref(self, expr: ast.expr, rel: str, cls: str | None) -> str | None:
        if isinstance(expr, ast.Name):
            found = self.target(expr.id, rel)
            return found[1] if found and found[0] == "sym" else None
        if not isinstance(expr, ast.Attribute):
            return None
        base, attr = expr.value, expr.attr
        if cls and isinstance(base, ast.Name) and base.id in ("self", "cls"):
            return self.method(cls, attr)
        is_super = isinstance(base, ast.Call) and isinstance(base.func, ast.Name)
        if cls and is_super and base.func.id == "super":
            return self.method(cls, attr, include_self=False)
        dotted = _dotted(base)
        found = self.target(dotted, rel) if dotted else None
        if not found:
            return None
        if found[0] == "mod":
            member = self.member(found[1], attr)
            return member[1] if member and member[0] == "sym" else None
        if self.nodes[found[1]].type == "Class":
            return self.method(found[1], attr)
        return None

    def link_bases(self) -> None:
        """INHERITS: класс -> базовый класс из репозитория, найденный по имени через импорты.
        Базы из внешних библиотек и динамически созданные базы пропускаются."""
        for rel, sid, node in self.classes:
            ids: list[str] = []
            for base in node.bases:
                if isinstance(base, ast.Subscript):
                    base = base.value
                found = self.ref(base, rel, None)
                if found and found != sid and self.nodes[found].type == "Class":
                    ids.append(found)
                    self.add_edge(sid, found, "INHERITS")
            self.bases[sid] = ids

    def link_calls(self) -> None:
        """CALLS: приблизительные связи. Разрешаются только вызовы f(), module.f(), Class.m(),
        self.m()/cls.m() с поиском по базовым классам и super().m().
        Вызовы obj.m() с неизвестным типом obj не угадываются и пропускаются.
        Вызовы на уровне модуля относятся к узлу файла."""
        for rel, sid, node, cls in self.funcs:
            for call in _calls(node, True):
                found = self.ref(call.func, rel, cls)
                if found and found != sid:
                    self.add_edge(sid, found, "CALLS", call.lineno)
        for rel, tree in self.trees.items():
            for call in _calls(tree, False):
                found = self.ref(call.func, rel, None)
                if found:
                    self.add_edge(rel, found, "CALLS", call.lineno)

    def link_docs(self, root: Path) -> None:
        """DOCUMENTS: раздел .md/.rst -> символ, если в тексте есть его имя (в обратных кавычках,
        роли :func:`...` или директиве autoclass). Связь ставится только при однозначном
        совпадении имени. Id раздела "path::#N", N с нуля по порядку разделов."""
        by_qn: dict[str, list[str]] = defaultdict(list)
        by_last: dict[str, list[str]] = defaultdict(list)
        for node in self.nodes.values():
            if node.type in ("Class", "Function"):
                by_qn[node.name].append(node.id)
                by_last[node.name.rsplit(".", 1)[-1]].append(node.id)
        for suffix, splitter in ((".md", _md_sections), (".rst", _rst_sections)):
            for path, rel in _iter_files(root, suffix):
                try:
                    text = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
                for n, (start, end, title, body) in enumerate(splitter(text)):
                    did = f"{rel}::#{n}"
                    self.nodes[did] = Node(did, "Doc", title or rel, rel, start, end)
                    for ref in _doc_refs(body, suffix == ".rst"):
                        found = self.match_span(ref, by_qn, by_last)
                        if found:
                            self.add_edge(did, found, "DOCUMENTS")

    @staticmethod
    def match_span(
        span: str, by_qn: dict[str, list[str]], by_last: dict[str, list[str]]
    ) -> str | None:
        text = re.sub(r"\(.*\)$", "", span.strip())
        if not _REF.fullmatch(text):
            return None
        parts = text.split(".")
        for cand in (text, ".".join(parts[-2:])):
            ids = by_qn.get(cand)
            if ids:
                return ids[0] if len(ids) == 1 else None
        last = parts[-1]
        ids = by_last.get(last) if len(last) >= 3 else None
        return ids[0] if ids and len(ids) == 1 else None

    def result(self) -> tuple[list[Node], list[Edge]]:
        edges = [e for e in self.edges.values() if e.src in self.nodes and e.dst in self.nodes]
        return list(self.nodes.values()), edges


def _build(root: Path) -> _Builder:
    builder = _Builder()
    for path, rel in _iter_files(root, ".py"):
        builder.add_module(path, rel)
    builder.index_modules()
    for rel, tree in builder.trees.items():
        builder.collect_defs(rel, tree)
        builder.collect_imports(rel, tree)
    builder.resolve_imports()
    builder.link_imports()
    builder.link_bases()
    builder.link_calls()
    builder.link_docs(root)
    return builder


def extract(root: Path) -> tuple[list[Node], list[Edge]]:
    """Узлы File/Class/Function/Doc и рёбра DEFINES/IMPORTS/INHERITS/CALLS/DOCUMENTS."""
    return _build(root).result()


def call_sites(root: Path) -> dict[tuple[str, str], int]:
    """Строка первого вызова для каждого ребра CALLS (src, dst): нужна для ручной проверки."""
    return _build(root).call_lines