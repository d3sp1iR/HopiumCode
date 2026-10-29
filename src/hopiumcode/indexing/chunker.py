"""Разбиение файлов на фрагменты. Владелец: участник 2.

Python через ast: каждая функция, метод и класс отдельным фрагментом (с декораторами и docstring),
верхнеуровневый код модуля фрагментом kind="module" (по одному на каждый непрерывный участок между
определениями, чтобы диапазон строк не накрывал чужой код). В тексте класса нет тел методов и вложенных
классов: они идут своими фрагментами, в модуле нет определений. Функции внутри функций остаются в тексте
родителя. Определения внутри if/try/with на уровне модуля или класса считаются определениями этого уровня.

Фрагмент длиннее MAX_TOKENS (слов * 1.5) режется по пустым строкам с перекрытием OVERLAP строк; номера
строк сохраняются. id: "путь::qualified_name" у первой части, а у следующих частей, кода модуля и повторных
имён (например, @overload или getter/setter) "путь::#N"; id уникальны в пределах файла.
"""
from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from pathlib import Path

from ..models import Chunk, ChunkKind

log = logging.getLogger("hopiumcode")

MAX_TOKENS = 300
OVERLAP = 2
_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_COMPOUND = (ast.If, ast.Try, ast.With, ast.AsyncWith, ast.For, ast.AsyncFor, ast.While)
if hasattr(ast, "TryStar"):
    _COMPOUND += (ast.TryStar,)


def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.5)


@dataclass
class _Unit:
    kind: ChunkKind
    name: str | None
    lines: list[int]  # номера строк (с 1), входящих во фрагмент, по возрастанию


def _span(node: ast.AST) -> range:
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    return range(start, node.end_lineno + 1)


def _defs_in(body: list[ast.stmt]) -> list[ast.AST]:
    """Определения этого уровня, включая спрятанные в if/try/with/for/while."""
    found = []
    for stmt in body:
        if isinstance(stmt, _DEFS):
            found.append(stmt)
        elif isinstance(stmt, _COMPOUND):
            for field in ("body", "orelse", "finalbody"):
                found += _defs_in(getattr(stmt, field, []) or [])
            for handler in getattr(stmt, "handlers", []) or []:
                found += _defs_in(handler.body)
    return found


def _without(lines: range | list[int], defs: list[ast.AST]) -> list[int]:
    excluded = {n for d in defs for n in _span(d)}
    return [n for n in lines if n not in excluded]


def _units(defs: list[ast.AST], prefix: str, in_class: bool) -> list[_Unit]:
    out = []
    for node in defs:
        qual = f"{prefix}{node.name}"
        if isinstance(node, ast.ClassDef):
            children = _defs_in(node.body)
            out.append(_Unit("class", qual, _without(_span(node), children)))
            out += _units(children, qual + ".", in_class=True)
        else:
            out.append(_Unit("method" if in_class else "function", qual, list(_span(node))))
    return out


def _strip_blank(lines: list[int], src: list[str]) -> list[int]:
    i, j = 0, len(lines)
    while i < j and not src[lines[i] - 1].strip():
        i += 1
    while j > i and not src[lines[j - 1] - 1].strip():
        j -= 1
    return lines[i:j]


def _text(lines: list[int], src: list[str]) -> str:
    return "\n".join(src[n - 1] for n in lines)


def _segments(lines: list[int], src: list[str]) -> list[list[int]]:
    """Абзацы (между пустыми строками и разрывами нумерации), каждый не длиннее MAX_TOKENS."""
    paragraphs, cur = [], []
    for n in lines:
        blank = not src[n - 1].strip()
        if cur and (blank or n != cur[-1] + 1):
            paragraphs.append(cur)
            cur = []
        if not blank:
            cur.append(n)
    if cur:
        paragraphs.append(cur)
    segments = []
    for par in paragraphs:
        if estimate_tokens(_text(par, src)) <= MAX_TOKENS:
            segments.append(par)
            continue
        part: list[int] = []
        for n in par:  # абзац без пустых строк длиннее лимита: режем по строкам
            if part and estimate_tokens(_text(part + [n], src)) > MAX_TOKENS:
                segments.append(part)
                part = []
            part.append(n)
        if part:
            segments.append(part)
    return segments


def _runs(lines: list[int]) -> list[list[int]]:
    """Непрерывные участки: разрыв нумерации означает вырезанное определение."""
    runs: list[list[int]] = []
    for n in lines:
        if runs and n == runs[-1][-1] + 1:
            runs[-1].append(n)
        else:
            runs.append([n])
    return runs


def _split_run(lines: list[int], src: list[str]) -> list[list[int]]:
    lines = _strip_blank(lines, src)
    if not lines or estimate_tokens(_text(lines, src)) <= MAX_TOKENS:
        return [lines] if lines else []
    pieces: list[list[int]] = []
    cur: list[int] = []
    fresh = 0  # сколько строк текущей части не взято из перекрытия
    for seg in _segments(lines, src):
        if fresh and estimate_tokens(_text(cur + seg, src)) > MAX_TOKENS:
            pieces.append(cur)
            cur = cur[-OVERLAP:] if len(cur) > OVERLAP else []
            fresh = 0
        cur = cur + seg
        fresh += len(seg)
    if fresh:
        pieces.append(cur)
    return pieces


def split_lines(lines: list[int], src: list[str]) -> list[list[int]]:
    """Режет фрагмент на непрерывные части не длиннее ~MAX_TOKENS.

    Части не перешагивают вырезанные определения (иначе диапазон строк фрагмента накрыл бы чужой код),
    а внутри непрерывного участка каждая следующая часть начинается с OVERLAP строк предыдущей.
    """
    return [piece for run in _runs(lines) for piece in _split_run(run, src)]


def chunk_source(source: str, rel_path: str) -> list[Chunk]:
    """Фрагменты Python-кода; rel_path пишется в Chunk.path и id. SyntaxError пробрасывается."""
    tree = ast.parse(source, filename=rel_path)
    src = source.splitlines()
    top = _defs_in(tree.body)
    units = [_Unit("module", None, _without(range(1, len(src) + 1), top))] + _units(top, "", in_class=False)

    chunks: list[Chunk] = []
    seen: set[str] = set()
    counter = 0
    for u in units:
        pieces = split_lines(u.lines, src)
        for i, piece in enumerate(pieces):
            cid = f"{rel_path}::{u.name}" if u.name and i == 0 else None  # остальные части: #N
            if cid is None or cid in seen:
                counter += 1
                cid = f"{rel_path}::#{counter}"
            seen.add(cid)
            text = _text(list(range(piece[0], piece[-1] + 1)), src)  # часть непрерывна: текст = строки файла
            chunks.append(Chunk(cid, rel_path, piece[0], piece[-1], u.kind, u.name, text, "python"))
    chunks.sort(key=lambda c: (c.start_line, -c.end_line))
    return chunks


def chunk_file(path: Path, root: Path) -> list[Chunk]:
    """Python: функции, методы, классы и код модуля через ast.

    Индексируются только .py (решение команды, docs/decisions.md). Файл с синтаксической ошибкой или
    не в UTF-8 пропускается с предупреждением в журнал.
    """
    path, root = Path(path), Path(root)
    rel = path.relative_to(root).as_posix()
    try:
        source = path.read_text(encoding="utf-8")
        return chunk_source(source, rel)
    except (SyntaxError, UnicodeDecodeError, ValueError) as ex:
        log.warning("Пропущен %s: %s", rel, ex)
        return []
