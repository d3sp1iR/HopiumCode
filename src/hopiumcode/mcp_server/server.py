"""MCP-сервер, транспорт stdio, инструменты только на чтение. Владелец: участник 4.

Инструменты: read_file, search_code, find_callers, git_log, git_blame.
Безопасность: пути только внутри REPO_ROOT, лимит ответа 8000 символов, git без shell.

Запуск: python -m hopiumcode.mcp_server.server (корень: переменная REPO_ROOT, иначе Config.repo_root).
SDK: пакет mcp 2.x, класс MCPServer (в mcp 1.x он назывался FastMCP).
Каждый инструмент — обычная функция с явным root: её можно тестировать без MCP. Ошибки возвращаются
текстом «Ошибка: ...», а не исключением, чтобы модель могла прочитать причину.
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from pydantic import Field

SERVER_NAME = "hopiumcode"
MAX_CHARS = 8000
DEFAULT_LINES = 200
MAX_LINES = 400
GIT_TIMEOUT = 10
MAX_K = 20
MAX_DEPTH = 3
MAX_CALLERS = 50
MAX_LOG = 50
READ_SUFFIXES = {".py"}  # решение команды: в индексе и ссылках только .py

# Тексты для модели: описания инструментов и аргументов (правятся здесь, см. docs/mcp_tools.md)
DESCRIPTIONS = {
    "search_code": (
        "Вызывай первым для вопросов «где и как это работает», «за что отвечает модуль», «из чего состоит»: "
        "не отвечай по памяти и не пиши вызов текстом, а вызови, например "
        'search_code(query="маршрутизация URL route add_url_rule"). '
        "Не подходит для «кто вызывает» (find_callers) и «кто менял, в каком коммите» (git_log, git_blame). "
        'Результат: строки "[путь:начало-конец] | вид | имя | начало кода".'
    ),
    "read_file": (
        "Вызывай, когда путь к файлу уже известен из вопроса или другого инструмента и нужно увидеть код, "
        'например read_file(path="app.py", start_line=900, end_line=960). '
        "Не ищи им, где что находится: для этого search_code. "
        'Результат: первая строка "[путь:начало-конец]", затем строки "  12| код".'
    ),
    "find_callers": (
        "Вызывай для вопросов «кто вызывает X», «где используется X», «что сломается, если изменить X»: передай "
        'имя функции из вопроса, например find_callers(symbol="make_response"). '
        "Не подходит для поиска по смыслу (search_code) и для истории (git_log). "
        'Результат: строки "[путь:начало-конец] | имя вызывающей функции".'
    ),
    "git_log": (
        "Вызывай для вопросов «кто и когда менял файл», «последний коммит», «история изменений», например "
        'git_log(path="ctx.py", n=5). '
        "Для одной строки или функции («кто добавил метод») — git_blame. "
        'Результат: строки "хэш | автор | дата | сообщение коммита".'
    ),
    "git_blame": (
        "Вызывай для вопросов «кто и в каком коммите добавил или изменил функцию, строку», например "
        'git_blame(path="sessions.py", line=215); если номер строки неизвестен, сначала найди его через search_code. '
        "Для истории всего файла — git_log. "
        'Результат: "[путь:строка-строка] коммит хэш | автор | дата | сообщение" и сама строка.'
    ),
}
ARGS = {
    "path": "Путь к файлу от корня кода (src/flask), например ctx.py или sansio/scaffold.py.",
    "start_line": "Первая строка (с 1). По умолчанию 1.",
    "end_line": f"Последняя строка включительно. По умолчанию start_line + {DEFAULT_LINES - 1}; "
                f"за один вызов не больше {MAX_LINES} строк.",
    "n": f"Сколько последних коммитов показать (1-{MAX_LOG}), по умолчанию 5.",
    "line": "Номер строки файла (с 1).",
    "query": "Что ищем: вопрос или ключевые слова, лучше с английскими именами из кода.",
    "k": f"Сколько фрагментов вернуть (1-{MAX_K}), по умолчанию 5.",
    "symbol": "Имя функции или метода, например make_response или Flask.make_response.",
    "depth": f"Глубина: 1 — прямые вызывающие, 2 — и их вызывающие (до {MAX_DEPTH}).",
}


class ToolInputError(ValueError):
    """Ошибка в аргументах инструмента; текст показывается модели."""


def get_root() -> Path:
    """Корень репозитория: REPO_ROOT из окружения или Config.repo_root."""
    env = os.environ.get("REPO_ROOT")
    if env:
        return Path(env).resolve()
    from ..config import Config
    return Path(Config.from_env().repo_root).resolve()


def resolve_path(root: Path, path: str) -> Path:
    """Путь внутри root. Запрещены пустой, абсолютный путь, "..", "~" и симлинки, ведущие наружу."""
    raw = (path or "").strip().replace("\\", "/")
    if not raw:
        raise ToolInputError("не указан путь")
    rel = Path(raw)
    if rel.is_absolute() or raw.startswith("~") or ".." in rel.parts:
        raise ToolInputError(f"путь вне репозитория: {path} (нужен путь относительно корня, без '..')")
    full = (root / rel).resolve()
    if not full.is_relative_to(root):
        raise ToolInputError(f"путь вне репозитория: {path}")
    return full


def locate(root: Path, path: str) -> Path:
    """resolve_path, а если файла нет — пробует отрезать префикс, совпадающий с хвостом корня.

    Модель часто передаёт путь от корня git (src/flask/ctx.py), хотя корень инструментов — src/flask.
    """
    full = resolve_path(root, path)
    if full.exists():
        return full
    raw = path.strip().replace("\\", "/")
    for i in range(1, len(root.parts)):
        prefix = "/".join(root.parts[-i:]) + "/"
        if raw.startswith(prefix):
            candidate = resolve_path(root, raw[len(prefix):])
            if candidate.exists():
                return candidate
    return full


def _rel(root: Path, full: Path) -> str:
    return full.relative_to(root).as_posix()


def _error(text: str) -> str:
    return f"Ошибка: {text}"


def read_file(root: Path, path: str, start_line: int | None = None, end_line: int | None = None) -> str:
    """Строки файла в формате "  12| текст"; первая строка ответа — ссылка [путь:начало-конец]."""
    root = Path(root).resolve()
    try:
        full = locate(root, path)
    except ToolInputError as ex:
        return _error(str(ex))
    if not full.exists():
        return _error(f"файл не найден: {path}. Найдите путь через search_code.")
    if full.is_dir():
        return _error(f"{path} — каталог, а не файл")
    if full.suffix not in READ_SUFFIXES:
        return _error(f"читать можно только файлы {', '.join(sorted(READ_SUFFIXES))}: {path}")
    try:
        lines = full.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return _error(f"файл не в UTF-8: {path}")
    total = len(lines)
    rel = _rel(root, full)
    if total == 0:
        return f"[{rel}] пустой файл"

    start = max(1, start_line or 1)
    if start > total:
        return _error(f"в {rel} всего {total} строк, start_line={start} за концом файла")
    end = end_line if end_line is not None else start + DEFAULT_LINES - 1
    if end < start:
        return _error(f"start_line ({start}) больше end_line ({end})")
    notes = []
    end = min(end, total)
    if end - start + 1 > MAX_LINES:
        end = start + MAX_LINES - 1
        notes.append(f"за один вызов не больше {MAX_LINES} строк")

    budget = MAX_CHARS - 300  # запас на заголовок и пометку
    body: list[str] = []
    used = 0
    for n in range(start, end + 1):
        line = f"{n:>4}| {lines[n - 1]}"
        if body and used + len(line) + 1 > budget:
            notes.append(f"обрезано по лимиту {MAX_CHARS} символов")
            break
        body.append(line[:budget])
        used += len(line) + 1
    shown_end = start + len(body) - 1
    header = f"[{rel}:{start}-{shown_end}] из {total} строк"
    footer = ""
    if notes:
        footer = f"\n… {'; '.join(notes)}. Продолжение: read_file(path={rel!r}, start_line={shown_end + 1})"
    elif shown_end < total:
        footer = f"\n… дальше строки {shown_end + 1}-{total}: read_file(path={rel!r}, start_line={shown_end + 1})"
    return f"{header}\n" + "\n".join(body) + footer


# --------------------------------------------------------------------------- git

def _git(root: Path, args: list[str]) -> subprocess.CompletedProcess:
    """git без shell, cwd=root (git сам найдёт репозиторий выше), таймаут GIT_TIMEOUT."""
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"}
    return subprocess.run(["git", "-c", "core.quotepath=off", *args], cwd=root, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=GIT_TIMEOUT, check=False, env=env)


def _git_failure(ex_or_res, root: Path) -> str:
    if isinstance(ex_or_res, subprocess.TimeoutExpired):
        return _error(f"git не ответил за {GIT_TIMEOUT} c")
    if isinstance(ex_or_res, FileNotFoundError):
        return _error("git не установлен")
    err = (ex_or_res.stderr or "").strip()
    if "not a git repository" in err:
        return _error(f"{root} не находится в git-репозитории")
    return _error(f"git: {err.splitlines()[-1] if err else 'код ' + str(ex_or_res.returncode)}")


def _git_target(root: Path, path: str) -> tuple[Path, str]:
    full = locate(root, path)
    if not full.is_file():
        raise ToolInputError(f"файл не найден: {path}")
    return full, _rel(root, full)


def git_log(root: Path, path: str, n: int = 5) -> str:
    """Строки "хэш7 | автор | дата ISO | сообщение (первая строка)", новые сверху."""
    root = Path(root).resolve()
    try:
        _, rel = _git_target(root, path)
    except ToolInputError as ex:
        return _error(str(ex))
    n = max(1, min(int(n or 5), MAX_LOG))
    try:
        res = _git(root, ["log", f"-n{n}", "--format=%H%x1f%an%x1f%aI%x1f%s", "--", rel])
    except (subprocess.TimeoutExpired, FileNotFoundError) as ex:
        return _git_failure(ex, root)
    if res.returncode != 0:
        return _git_failure(res, root)
    rows = []
    for line in res.stdout.splitlines():
        sha, author, date, subject = (line.split("\x1f") + ["", "", ""])[:4]
        rows.append(f"{sha[:7]} | {author} | {date} | {subject}")
    if not rows:
        return f"Коммитов, менявших {rel}, нет (файл не в истории git)."
    return f"git log {rel}, последние {len(rows)}:\n" + "\n".join(rows)


def _parse_porcelain(text: str) -> dict:
    lines = text.splitlines()
    info = {"sha": lines[0].split()[0]}
    for line in lines[1:]:
        if line.startswith("\t"):
            info["content"] = line[1:]
            break
        key, _, value = line.partition(" ")
        info[key] = value
    return info


def _iso(epoch: str, tz: str) -> str:
    sign = -1 if tz.startswith("-") else 1
    offset = timedelta(hours=int(tz[1:3]), minutes=int(tz[3:5])) * sign
    return datetime.fromtimestamp(int(epoch), timezone(offset)).isoformat()


def git_blame(root: Path, path: str, line: int) -> str:
    """Коммит, автор, дата и сообщение для строки (git blame -L line,line --porcelain)."""
    root = Path(root).resolve()
    try:
        full, rel = _git_target(root, path)
    except ToolInputError as ex:
        return _error(str(ex))
    try:
        total = len(full.read_text(encoding="utf-8", errors="replace").splitlines())
    except OSError as ex:
        return _error(f"не удалось прочитать {rel}: {ex}")
    if not isinstance(line, int) or line < 1 or line > total:
        return _error(f"в {rel} {total} строк, строка {line} вне диапазона")
    try:
        res = _git(root, ["blame", "-L", f"{line},{line}", "--porcelain", "--", rel])
    except (subprocess.TimeoutExpired, FileNotFoundError) as ex:
        return _git_failure(ex, root)
    if res.returncode != 0:
        return _git_failure(res, root)
    info = _parse_porcelain(res.stdout)
    ref = f"[{rel}:{line}-{line}]"
    content = info.get("content", "")
    if set(info["sha"]) == {"0"}:
        return f"{ref} строка ещё не закоммичена\n{line:>4}| {content}"
    date = _iso(info["author-time"], info.get("author-tz", "+0000")) if "author-time" in info else "?"
    return (f"{ref} коммит {info['sha'][:7]} | {info.get('author', '?')} | {date} | {info.get('summary', '')}\n"
            f"{line:>4}| {content}")


# --------------------------------------------------------------------------- поиск и граф (участники 2 и 3)

def _warn(text: str) -> str:
    print(f"⚠ {text}", file=sys.stderr)
    return f"⚠ {text}"


def _short(text: str, limit: int = 120) -> str:
    one = " ".join(text.split())
    return one if len(one) <= limit else one[:limit - 1] + "…"


def _fit(lines: list[str]) -> str:
    out, used = [], 0
    for line in lines:
        if used + len(line) + 1 > MAX_CHARS - 60:
            out.append(f"… обрезано по лимиту {MAX_CHARS} символов")
            break
        out.append(line)
        used += len(line) + 1
    return "\n".join(out)


def search_code(root: Path, query: str, k: int = 5) -> str:
    """Фрагменты через retrieval.hybrid.retrieve: "[путь:начало-конец] | kind | имя | первые 120 символов"."""
    if not (query or "").strip():
        return _error("пустой запрос")
    k = max(1, min(int(k or 5), MAX_K))
    from ..retrieval.hybrid import retrieve
    header = f"search_code {query!r}, найдено фрагментов: {{n}}"
    try:
        hits = retrieve(query, k, "hybrid")
        stub = False
    except NotImplementedError as ex:
        from ..stubs import stub_retrieve
        hits = stub_retrieve(query, k, "hybrid")
        stub = True
        header = _warn(f"ЗАГЛУШКА: retrieval.hybrid.retrieve не реализован ({ex}); ниже демо-фрагменты не из "
                       "репозитория, ссылаться на них нельзя") + "\n" + header
    if not hits:
        return header.format(n=0) + "\nНичего не найдено. Уточните запрос именами из кода."
    rows = []
    for h in hits[:k]:
        c = h.chunk
        ref = f"{c.path}:{c.start_line}-{c.end_line}"
        ref = ref if stub else f"[{ref}]"  # демо-ссылки без скобок: оркестратор не примет их как источник
        rows.append(f"{ref} | {c.kind} | {c.name or 'module'} | {_short(c.text)}")
    return _fit([header.format(n=len(rows)), *rows])


def _defs(tree: ast.AST) -> list[tuple[int, int, str]]:
    """(начало с декораторами, конец, qualified name) всех функций и методов файла."""
    out = []

    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                walk(child, f"{prefix}{child.name}.")
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = min([child.lineno] + [d.lineno for d in child.decorator_list])
                out.append((start, child.end_lineno, f"{prefix}{child.name}"))
                walk(child, f"{prefix}{child.name}.")
            else:
                walk(child, prefix)
    walk(tree, "")
    return out


def _py_files(root: Path):
    for p in sorted(root.rglob("*.py")):
        rel = p.relative_to(root)
        if "__pycache__" in rel.parts or any(part.startswith(".") for part in rel.parts):
            continue
        if p.resolve().is_relative_to(root):
            yield p


def _parse(path: Path):
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return None


def _locate(root: Path, rel: str, qualname: str) -> tuple[int, int] | None:
    try:
        full = resolve_path(root, rel)
    except ToolInputError:
        return None
    tree = _parse(full) if full.is_file() else None
    for start, end, name in _defs(tree) if tree else []:
        if name == qualname:
            return start, end
    return None


def _text_callers(root: Path, name: str) -> list[tuple[str, int, int, str]]:
    """Функции, в теле которых встречается вызов name( — эвристика вместо графа."""
    call = re.compile(rf"(?<![\w]){re.escape(name)}\s*\(")
    own_def = re.compile(rf"^\s*(?:async\s+)?def\s+{re.escape(name)}\b")
    found: dict[tuple[str, str], tuple[int, int]] = {}
    for path in _py_files(root):
        text = path.read_text(encoding="utf-8", errors="replace")
        if name not in text:
            continue
        tree = _parse(path)
        defs = _defs(tree) if tree else []
        rel = _rel(root, path)
        for n, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith("#") or own_def.match(line) or not call.search(line):
                continue
            inner = [d for d in defs if d[0] <= n <= d[1]]
            if inner:
                start, end, qual = max(inner, key=lambda d: d[0])  # самая вложенная функция
            else:
                start, end, qual = n, n, "<код модуля>"
            found.setdefault((rel, qual), (start, end))
    return [(rel, s, e, q) for (rel, q), (s, e) in sorted(found.items(), key=lambda x: (x[0][0], x[1][0]))]


def _graph_callers(root: Path, symbol: str, depth: int) -> list[str]:
    from ..graph.query import callers
    return list(callers(symbol, depth))


def find_callers(root: Path, symbol: str, depth: int = 1) -> str:
    """Вызывающие функции: "[путь:начало-конец] | имя вызывающей функции"."""
    root = Path(root).resolve()
    symbol = (symbol or "").strip()
    if not re.fullmatch(r"[A-Za-z_][\w.]*", symbol):
        return _error(f"нужно имя функции или метода, например make_response, а не {symbol!r}")
    depth = max(1, min(int(depth or 1), MAX_DEPTH))
    try:
        ids = _graph_callers(root, symbol, depth)
    except NotImplementedError as ex:
        return _find_callers_by_text(root, symbol, depth, ex)
    rows = []
    for node_id in ids[:MAX_CALLERS]:
        rel, sep, qual = node_id.partition("::")
        span = _locate(root, rel, qual) if sep else None
        rows.append(f"[{rel}:{span[0]}-{span[1]}] | {qual}" if span else f"{node_id} | (строки не найдены)")
    if not rows:
        return f"Граф: вызывающих {symbol} не найдено."
    return _fit([f"find_callers {symbol} (граф кода, глубина {depth}), найдено: {len(rows)}", *rows])


def _find_callers_by_text(root: Path, symbol: str, depth: int, ex: Exception) -> str:
    warn = _warn(f"граф кода не реализован ({ex}): вызывающие найдены текстовым поиском '{symbol.split('.')[-1]}(' "
                 "по .py-файлам, возможны ложные совпадения и пропуски")
    rows, seen = [], set()
    frontier = [symbol.split(".")[-1]]
    for level in range(1, depth + 1):
        nxt = []
        for name in frontier:
            for rel, start, end, qual in _text_callers(root, name):
                if (rel, qual) in seen:
                    continue
                seen.add((rel, qual))
                suffix = f" (уровень {level}, вызывает {name})" if depth > 1 else ""
                rows.append(f"[{rel}:{start}-{end}] | {qual}{suffix}")
                if qual != "<код модуля>":
                    nxt.append(qual.split(".")[-1])
        frontier = nxt
    if not rows:
        return f"{warn}\nВызовов {symbol} не найдено."
    return _fit([warn, f"find_callers {symbol}, найдено: {len(rows)}", *rows[:MAX_CALLERS]])


def create_server(root: Path | None = None) -> MCPServer:
    root = Path(root or get_root()).resolve()
    server = MCPServer(SERVER_NAME, instructions=f"Инструменты чтения кода репозитория (корень {root.name}).",
                       log_level="WARNING")

    @server.tool(name="read_file", description=DESCRIPTIONS["read_file"])
    def _read_file(path: Annotated[str, Field(description=ARGS["path"])],
                   start_line: Annotated[int | None, Field(description=ARGS["start_line"])] = None,
                   end_line: Annotated[int | None, Field(description=ARGS["end_line"])] = None) -> str:
        return read_file(root, path, start_line, end_line)

    @server.tool(name="git_log", description=DESCRIPTIONS["git_log"])
    def _git_log(path: Annotated[str, Field(description=ARGS["path"])],
                 n: Annotated[int, Field(description=ARGS["n"])] = 5) -> str:
        return git_log(root, path, n)

    @server.tool(name="git_blame", description=DESCRIPTIONS["git_blame"])
    def _git_blame(path: Annotated[str, Field(description=ARGS["path"])],
                   line: Annotated[int, Field(description=ARGS["line"])]) -> str:
        return git_blame(root, path, line)

    @server.tool(name="search_code", description=DESCRIPTIONS["search_code"])
    def _search_code(query: Annotated[str, Field(description=ARGS["query"])],
                     k: Annotated[int, Field(description=ARGS["k"])] = 5) -> str:
        return search_code(root, query, k)

    @server.tool(name="find_callers", description=DESCRIPTIONS["find_callers"])
    def _find_callers(symbol: Annotated[str, Field(description=ARGS["symbol"])],
                      depth: Annotated[int, Field(description=ARGS["depth"])] = 1) -> str:
        return find_callers(root, symbol, depth)

    return server


def main() -> None:
    create_server().run("stdio")


if __name__ == "__main__":
    main()
