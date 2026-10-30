"""Разбиение файлов на фрагменты. Владелец: участник 2.

Идентификаторы совпадают с узлами графа (graph/extract.py), чтобы обход графа находил фрагменты:
  функция, метод, класс -> "path::Class.method" (как Node.id);
  код верхнего уровня модуля -> "path" (как узел File);
  раздел Markdown/rst -> "path::#N" (N считается теми же функциями, что и в графе);
  .txt -> "path::#0".
Длинный фрагмент режется на части: первая сохраняет id, остальные получают суффикс "~2", "~3", ...
Повторное определение того же имени (property.setter, ветки if/else) получает суффикс "@2", "@3": в графе
такие определения сливаются в один узел, а для поиска нужны оба текста.
"""
from __future__ import annotations

import ast
import logging
import math
from pathlib import Path

from ..graph.extract import _blocks, _md_sections, _rst_sections
from ..models import Chunk

log = logging.getLogger(__name__)

MAX_TOKENS = 300   # оценка: слов * 1.5
OVERLAP_LINES = 2

_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text.split()) * 1.5)


def split_lines(lines: list[str], first_line: int, max_tokens: int = MAX_TOKENS,
                overlap: int = OVERLAP_LINES) -> list[tuple[int, int, str]]:
    """Режет строки на части не длиннее max_tokens: по возможности по пустой строке, с перекрытием.

    Возвращает (первая строка, последняя строка, текст); номера строк сквозные, считая от first_line.
    Одна строка длиннее лимита остаётся целой частью.
    """
    costs = [estimate_tokens(line) for line in lines]
    if sum(costs) <= max_tokens:
        return [(first_line, first_line + len(lines) - 1, "\n".join(lines))] if lines else []
    parts: list[tuple[int, int, str]] = []
    i, n = 0, len(lines)
    while i < n:
        j, total = i, 0
        while j < n and (j == i or total + costs[j] <= max_tokens):
            total += costs[j]
            j += 1
        if j < n:
            for k in range(j - 1, i + (j - i) // 2, -1):    # пустая строка во второй половине части
                if not lines[k].strip():
                    j = k + 1
                    break
        piece = lines[i:j]
        while piece and not piece[-1].strip():
            piece.pop()
        if any(line.strip() for line in piece):
            parts.append((first_line + i, first_line + i + len(piece) - 1, "\n".join(piece)))
        if j >= n:
            break
        i = max(j - overlap, i + 1)
    return parts


def _emit(out: list[Chunk], base_id: str, path: str, first_line: int, lines: list[str],
          kind: str, name: str | None, lang: str) -> None:
    for n, (start, end, text) in enumerate(split_lines(lines, first_line), start=1):
        out.append(Chunk(base_id if n == 1 else f"{base_id}~{n}", path, start, end,
                         kind, name, text, lang))  # type: ignore[arg-type]


def _def_start(node: ast.AST) -> int:
    decorators = getattr(node, "decorator_list", [])
    return min([node.lineno, *(d.lineno for d in decorators)])  # type: ignore[attr-defined]


def _direct_defs(body: list[ast.stmt]) -> list[ast.AST]:
    """Определения этого уровня, включая лежащие внутри if/try/with (как в графе)."""
    found: list[ast.AST] = []
    for st in body:
        if isinstance(st, _DEFS):
            found.append(st)
        elif isinstance(st, (ast.If, ast.Try, ast.With)):
            for block in _blocks(st):
                found.extend(_direct_defs(block))
    return found


def _chunk_python(rel: str, text: str) -> list[Chunk]:
    tree = ast.parse(text, filename=rel)
    lines = text.splitlines()
    out: list[Chunk] = []
    covered: set[int] = set()
    seen: dict[str, int] = {}     # одно имя определено несколько раз (property.setter, if/else)

    def base_id(qn: str) -> str:
        seen[qn] = seen.get(qn, 0) + 1
        return f"{rel}::{qn}" if seen[qn] == 1 else f"{rel}::{qn}@{seen[qn]}"

    def visit(body: list[ast.stmt], prefix: str, in_class: bool, top: bool) -> None:
        for node in _direct_defs(body):
            qn = prefix + node.name  # type: ignore[attr-defined]
            cid = base_id(qn)
            start, end = _def_start(node), node.end_lineno or node.lineno  # type: ignore[attr-defined]
            if top:
                covered.update(range(start, end + 1))
            if isinstance(node, ast.ClassDef):
                inner = _direct_defs(node.body)
                head_end = min(_def_start(d) for d in inner) - 1 if inner else end
                head = lines[start - 1:max(head_end, start)]
                while len(head) > 1 and not head[-1].strip():
                    head.pop()
                _emit(out, cid, rel, start, head, "class", qn, "python")
                visit(node.body, qn + ".", True, False)
            else:
                kind = "method" if in_class else "function"
                _emit(out, cid, rel, start, lines[start - 1:end], kind, qn, "python")

    visit(tree.body, "", False, True)

    # Код верхнего уровня: всё, что не вошло в определения (импорты, константы, докстринг модуля).
    rest = [(n, line) for n, line in enumerate(lines, start=1) if n not in covered]
    while rest and not rest[0][1].strip():
        rest.pop(0)
    while rest and not rest[-1][1].strip():
        rest.pop()
    collapsed: list[tuple[int, str]] = []   # подряд идущие пустые строки схлопываем
    for n, line in rest:
        if line.strip() or (collapsed and collapsed[-1][1].strip()):
            collapsed.append((n, line))
    out.extend(_module_chunks(rel, collapsed))
    return out


def _module_chunks(rel: str, collapsed: list[tuple[int, str]]) -> list[Chunk]:
    """Фрагменты кода верхнего уровня. Номера строк берутся из исходного файла: диапазон ссылки
    охватывает первую и последнюю строку кода верхнего уровня (между ними могут лежать функции)."""
    texts = [line for _, line in collapsed]
    chunks: list[Chunk] = []
    for k, (a, b, part) in enumerate(split_lines(texts, 1), start=1):
        chunks.append(Chunk(rel if k == 1 else f"{rel}~{k}", rel, collapsed[a - 1][0],
                            collapsed[b - 1][0], "module", None, part, "python"))
    return chunks


def _chunk_sections(rel: str, text: str, suffix: str) -> list[Chunk]:
    lines = text.splitlines()
    if suffix == ".txt":
        sections = [(1, len(lines), "", text)] if text.strip() else []
        lang = "text"
    else:
        sections = (_md_sections if suffix == ".md" else _rst_sections)(text)
        lang = "markdown" if suffix == ".md" else "rst"
    out: list[Chunk] = []
    for n, (start, end, title, _body) in enumerate(sections):
        _emit(out, f"{rel}::#{n}", rel, start, lines[start - 1:end], "doc", title or None, lang)
    return out


def chunk_file(path: Path, root: Path) -> list[Chunk]:
    """Python: функции, методы, классы через ast; Markdown/rst: по заголовкам; .txt: целиком.

    Файл с синтаксической ошибкой или не в UTF-8 пропускается с предупреждением (пустой список).
    """
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        rel = path.resolve().relative_to(root.resolve()).as_posix()
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as exc:
        log.warning("файл пропущен %s: %s", rel, exc)
        return []
    suffix = path.suffix.lower()
    if suffix == ".py":
        try:
            return _chunk_python(rel, text)
        except (SyntaxError, ValueError) as exc:
            log.warning("файл пропущен %s: %s", rel, exc)
            return []
    if suffix in (".md", ".rst", ".txt"):
        return _chunk_sections(rel, text, suffix)
    return []
