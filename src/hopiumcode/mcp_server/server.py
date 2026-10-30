"""MCP-сервер HopiumCode (зона участника 4).

Инструменты: search_code, read_file, find_callers, git_log, git_blame.
Описания для модели лежат в descriptions.py.

Транспорт stdio. Запуск из корня проекта:

    REPO_ROOT=data/repo/flask/src/flask .venv/bin/python -m hopiumcode.mcp_server.server

Важно для stdio: stdout занят протоколом JSON-RPC, поэтому никаких print();
диагностика идёт только через logging в stderr.
"""

import ast
import functools
import logging
import os
import re
import stat
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path, PureWindowsPath
from typing import Annotated

from pydantic import Field

try:  # mcp >= 2.0: FastMCP переименован в MCPServer, API декораторов тот же
    from mcp.server import MCPServer as FastMCP
except ImportError:
    try:
        from mcp.server.mcpserver import MCPServer as FastMCP
    except ImportError:  # mcp 1.x
        from mcp.server.fastmcp import FastMCP

from hopiumcode.mcp_server.descriptions import ARG_DESCRIPTIONS, TOOL_DESCRIPTIONS

# --- Лимиты -----------------------------------------------------------------
MAX_CHARS = 8000            # максимальная длина ответа целиком, вместе с пометкой
MAX_LINES = 400             # максимум строк за один вызов
MAX_FILE_BYTES = 2_000_000  # файлы больше не читаем вовсе (защита памяти)
BINARY_SNIFF_BYTES = 8192   # где ищем нулевой байт, чтобы распознать бинарник
NOTE_RESERVE = 300          # запас символов под пометку «обрезано»
GIT_TIMEOUT = 10            # секунд на один вызов git
GIT_LOG_MAX = 50            # потолок для n в git_log
SEARCH_K_MAX = 10           # потолок для k в search_code
SNIPPET_LINES = 6           # сколько первых строк фрагмента показывает search_code
CALLERS_DEPTH_MAX = 3       # потолок для depth в find_callers

ROOT_ENV_VARS = ("REPO_ROOT", "CG_REPO_ROOT")  # CG_REPO_ROOT — как в .env.example

log = logging.getLogger("hopiumcode.mcp_server")


class ToolInputError(ValueError):
    """Ошибка инструмента: FastMCP превратит её в результат с isError=true,
    текст исключения клиент увидит как есть."""


ReadFileError = ToolInputError  # старое имя, на него ссылаются тесты read_file


# --- Безопасное разрешение пути ---------------------------------------------
def get_repo_root() -> Path:
    """Корень репозитория: REPO_ROOT, иначе CG_REPO_ROOT, иначе Config.repo_root (тот же, что у индекса)."""
    raw = next((os.environ[v] for v in ROOT_ENV_VARS if os.environ.get(v)), None)
    if not raw:
        from hopiumcode.config import Config
        default = Path(Config.from_env().repo_root)
        if not default.is_dir():
            raise ReadFileError(f"не задана переменная окружения REPO_ROOT, а каталога по умолчанию {default} "
                                "нет (выполните make fetch или задайте REPO_ROOT)")
        return default.resolve()
    root = Path(raw).expanduser().resolve()
    if not root.is_dir():
        raise ReadFileError("REPO_ROOT не указывает на существующий каталог")
    return root


def resolve_in_repo(path: str, root: Path) -> Path:
    """Проверка пути по ТЗ (раздел 10, п. 3): запрет «..» и абсолютных путей,
    затем (root / path).resolve() и проверка, что результат лежит внутри root.

    resolve() раскрывает все символические ссылки, поэтому ссылка, ведущая
    наружу, даёт путь вне root и отсекается той же проверкой. Всё это идёт
    ДО проверки существования: снаружи нельзя узнать, есть ли там файл.
    """
    if not path or "\x00" in path:
        raise ToolInputError("пустой или некорректный путь")
    # PureWindowsPath ловит и "/etc", и "\\etc", и "C:\\x", и "C:x" на любой ОС
    if Path(path).is_absolute() or PureWindowsPath(path).anchor:
        raise ToolInputError("путь вне репозитория: абсолютные пути запрещены, "
                             "укажите путь относительно корня репозитория")
    if ".." in re.split(r"[\\/]", path):
        raise ToolInputError("путь вне репозитория: «..» в пути запрещено")
    try:
        candidate = (root / path).resolve()
    except (OSError, RuntimeError) as exc:  # петля ссылок и т. п.
        raise ToolInputError("не удалось разрешить путь") from exc
    # is_relative_to сравнивает части пути, а не строки:
    # /repo_evil не считается вложенным в /repo
    if not candidate.is_relative_to(root):
        raise ToolInputError("путь вне репозитория")
    return candidate


def resolve_existing(path: str) -> tuple[Path, Path]:
    """Общая проверка для всех инструментов: путь внутри REPO_ROOT и существует."""
    root = get_repo_root()
    target = resolve_in_repo(path, root)
    if not target.exists():
        # Модель часто передаёт путь от корня git (src/flask/ctx.py), а корень инструментов — src/flask:
        # пробуем отрезать префикс, совпадающий с хвостом корня (проверка «внутри репозитория» та же).
        norm = path.strip().replace("\\", "/")
        for i in range(1, len(root.parts)):
            prefix = "/".join(root.parts[-i:]) + "/"
            if norm.startswith(prefix) and norm[len(prefix):]:
                candidate = resolve_in_repo(norm[len(prefix):], root)
                if candidate.exists():
                    return root, candidate
        raise ToolInputError(f"файл не найден: {path}")
    return root, target


def _read_text(target: Path, shown: str) -> str:
    if not target.exists():
        raise ReadFileError(f"файл не найден: {shown}")
    if not target.is_file():
        raise ReadFileError(f"это не файл: {shown}")
    if target.stat().st_size > MAX_FILE_BYTES:
        raise ReadFileError(f"файл слишком большой (лимит {MAX_FILE_BYTES} байт): {shown}")

    # Путь уже без ссылок; O_NOFOLLOW не даст подменить последний компонент
    # на ссылку между проверкой и открытием, O_NONBLOCK — зависнуть на FIFO.
    flags = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
             | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0))
    try:
        fd = os.open(target, flags)
    except OSError as exc:
        raise ReadFileError(f"не удалось открыть файл: {shown}") from exc
    with os.fdopen(fd, "rb") as f:
        if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
            raise ReadFileError(f"это не обычный файл: {shown}")
        data = f.read(MAX_FILE_BYTES + 1)

    if len(data) > MAX_FILE_BYTES:
        raise ReadFileError(f"файл слишком большой (лимит {MAX_FILE_BYTES} байт): {shown}")
    if b"\x00" in data[:BINARY_SNIFF_BYTES]:
        raise ReadFileError(f"бинарный файл, чтение не поддерживается: {shown}")
    return data.decode("utf-8", errors="replace").removeprefix("\ufeff")


def _split_lines(text: str) -> list[str]:
    # Как универсальные переводы строк в open(): \n, \r\n, \r.
    # Не splitlines(): он режет ещё и по \f, \v и т. п. и сбивает нумерацию.
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


# --- Основная логика (тестируется без MCP) ----------------------------------
def read_file_impl(path: str, start_line: int | None = None,
                   end_line: int | None = None) -> str:
    _, target = resolve_existing(path)
    lines = _split_lines(_read_text(target, path))
    total = len(lines)
    if total == 0:
        return "(пустой файл)"

    start = 1 if start_line is None else start_line
    if start < 1:
        raise ReadFileError("start_line должен быть >= 1")
    if start > total:
        raise ReadFileError(f"start_line={start} больше числа строк в файле ({total})")
    if end_line is not None and end_line < start:
        raise ReadFileError("end_line меньше start_line")

    end = total if end_line is None else min(end_line, total)
    cut_by_lines = end - start + 1 > MAX_LINES
    if cut_by_lines:
        end = start + MAX_LINES - 1

    width = len(str(end))
    budget = MAX_CHARS - NOTE_RESERVE
    rows: list[str] = []
    used = 0
    shown_end = start - 1
    cut_by_chars = False
    line_cut_mid = False

    for n in range(start, end + 1):
        row = f" {n:>{width}}|{lines[n - 1]}"
        cost = len(row) + (1 if rows else 0)  # +1 за "\n" между строками
        if used + cost > budget:
            cut_by_chars = True
            if not rows:  # одна строка длиннее лимита: отдаём её начало
                rows.append(row[:budget])
                shown_end = n
                line_cut_mid = True
            break
        rows.append(row)
        used += cost
        shown_end = n

    result = "\n".join(rows)
    if cut_by_lines or cut_by_chars:
        reason = (f"лимит {MAX_CHARS} символов" if cut_by_chars
                  else f"лимит {MAX_LINES} строк")
        note = f"\n… [обрезано: {reason}; показаны строки {start}-{shown_end} из {total}"
        if line_cut_mid:
            note += f"; строка {shown_end} показана не полностью"
        if shown_end < total:
            note += f"; продолжение: start_line={shown_end + 1}"
        result += note + "]"
    return result[:MAX_CHARS]  # страховка: жёсткий потолок



# --- общий лимит ответа ----------------------------------------------------------
def cap_text(text: str, limit: int | None = None) -> str:
    """Обрезка по целым строкам до лимита ответа (8000 символов) с пометкой."""
    limit = MAX_CHARS if limit is None else limit
    if len(text) <= limit:
        return text
    mark = "\n… [обрезано: лимит ответа]"
    cut = text[: limit - len(mark)]
    if "\n" in cut:
        cut = cut.rsplit("\n", 1)[0]
    return cut + mark


# --- search_code: обёртка над retrieve() участника 3 ------------------------------
def _retrieve(query: str, k: int) -> list:
    """retrieve(question, k, mode) -> list[Hit] из общего контракта (ТЗ, раздел 5).

    Импорт ленивый: сервер должен стартовать и отдавать read_file и git,
    даже пока индекс и граф не готовы.
    """
    from hopiumcode.retrieval.hybrid import retrieve
    return retrieve(query, k=k, mode="hybrid")


def search_code_impl(query: str, k: int = 5) -> str:
    query = (query or "").strip()
    if not query:
        raise ToolInputError("пустой запрос")
    if k < 1:
        raise ToolInputError("k должен быть >= 1")
    k = min(k, SEARCH_K_MAX)
    from hopiumcode.indexing.vectorstore import IndexNotBuilt
    try:
        hits = _retrieve(query, k)
    except IndexNotBuilt as exc:
        raise ToolInputError(f"индекс не построен ({exc})") from exc
    except (ImportError, NotImplementedError) as exc:
        raise ToolInputError("поиск ещё не готов: нет retrieval.hybrid.retrieve "
                             "(участник 3) или индекса (участник 2)") from exc
    except FileNotFoundError as exc:  # граф для режима hybrid ещё не построен
        raise ToolInputError(f"индекс не построен ({exc}); выполните make index") from exc

    blocks = []
    for hit in hits[:k]:
        c = hit.chunk
        title = c.name or c.kind
        head = f"[{c.path}:{c.start_line}-{c.end_line}] {title} ({hit.via}, {hit.score:.2f})"
        text_lines = c.text.splitlines()
        width = len(str(c.start_line + SNIPPET_LINES))
        body = [f" {c.start_line + i:>{width}}|{line}"
                for i, line in enumerate(text_lines[:SNIPPET_LINES])]
        if len(text_lines) > SNIPPET_LINES:
            body.append(f" … ещё {len(text_lines) - SNIPPET_LINES} строк: "
                        f"read_file(path=\"{c.path}\", start_line={c.start_line}, "
                        f"end_line={c.end_line})")
        blocks.append("\n".join([head, *body]))
    if not blocks:
        return f"ничего не найдено по запросу: {query}"
    return cap_text("\n\n".join(blocks))


# --- find_callers: обёртка над графом участника 3 ---------------------------------
@functools.lru_cache(maxsize=1)
def _get_graph():
    """Загруженный GraphStore (контракт: callers(symbol, depth) -> list[str] id).

    Граф загружает graph.query.get_store() участника 3 из Config.graph_dir (data/graph); если граф не построен,
    load() бросает FileNotFoundError, и find_callers просит выполнить make index.
    """
    from hopiumcode.graph.query import get_store
    return get_store()


def _symbol_lines(root: Path, rel_path: str, qualname: str) -> tuple[int, int] | None:
    """Строки определения Class.method в файле — по ast, т.к. callers() отдаёт только id."""
    try:
        target = resolve_in_repo(rel_path, root)
        tree = ast.parse(target.read_text(encoding="utf-8", errors="replace"))
    except (ToolInputError, OSError, SyntaxError, ValueError):
        return None
    nodes: list[ast.AST] = [tree]
    for part in qualname.split("."):
        found = None
        for parent in nodes:
            for node in ast.iter_child_nodes(parent):
                if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                        and node.name == part):
                    found = node
                    break
            if found:
                break
        if found is None:
            return None
        nodes = [found]
    node = nodes[0]
    start = min([node.lineno, *(d.lineno for d in getattr(node, "decorator_list", []))])
    return start, getattr(node, "end_lineno", node.lineno)


def _format_symbol_id(root: Path, symbol_id: str) -> str:
    """'path::Class.method' -> '[path:начало-конец] Class.method'."""
    path, sep, qualname = symbol_id.partition("::")
    if not sep:
        return symbol_id
    lines = _symbol_lines(root, path, qualname) if not qualname.startswith("#") else None
    where = f"{path}:{lines[0]}-{lines[1]}" if lines else path
    return f"[{where}] {qualname}"


def find_callers_impl(symbol: str, depth: int = 1) -> str:
    symbol = (symbol or "").strip().removesuffix("()")
    if not symbol:
        raise ToolInputError("пустое имя символа")
    if depth < 1:
        raise ToolInputError("depth должен быть >= 1")
    depth = min(depth, CALLERS_DEPTH_MAX)
    root = get_repo_root()
    try:
        graph = _get_graph()
    except (ImportError, NotImplementedError) as exc:
        raise ToolInputError("граф кода ещё не готов (graph/, участник 3)") from exc
    except FileNotFoundError as exc:
        raise ToolInputError(f"граф кода не построен ({exc}); выполните make index") from exc
    try:
        ids = graph.callers(symbol, depth=depth)
    except NotImplementedError as exc:
        raise ToolInputError("граф кода ещё не готов (graph/, участник 3)") from exc
    if not ids:
        return f"вызовов не найдено: {symbol}"
    return cap_text("\n".join(_format_symbol_id(root, i) for i in dict.fromkeys(ids)))


# --- git ---------------------------------------------------------------------
# Сообщения git разбираем по-английски, поэтому LC_ALL=C.
_GIT_ENV = {"LC_ALL": "C", "LANGUAGE": "C", "GIT_TERMINAL_PROMPT": "0"}
_ZERO_SHA = "0" * 40


def _git_error_text(stderr: str, shown: str) -> str:
    if "not a git repository" in stderr:
        return "REPO_ROOT не является git-репозиторием"
    if "dubious ownership" in stderr:
        return ("git отказался работать с репозиторием чужого владельца; "
                "выполните: git config --global --add safe.directory <путь>")
    if "does not have any commits yet" in stderr or "no such ref: HEAD" in stderr:
        return "в репозитории ещё нет коммитов"
    if m := re.search(r"has only (\d+) lines?", stderr):
        return f"строка вне диапазона: число строк в файле {shown} — {m.group(1)}"
    if "no such path" in stderr:
        return f"файл не отслеживается git: {shown}"
    first = stderr.strip().splitlines()[0] if stderr.strip() else "неизвестная ошибка"
    return f"ошибка git: {first.removeprefix('fatal: ')}"


def run_git(args: list[str], root: Path, shown: str) -> str:
    """git со списком аргументов, без shell, cwd=REPO_ROOT, с таймаутом."""
    cmd = ["git", "--no-pager", "-c", "core.fsmonitor=false", *args]
    try:
        proc = subprocess.run(
            cmd,
            cwd=root,
            capture_output=True,
            timeout=GIT_TIMEOUT,
            env={**os.environ, **_GIT_ENV},
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ToolInputError("git не установлен или не найден в PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise ToolInputError(f"git не ответил за {GIT_TIMEOUT} секунд") from exc
    stdout = proc.stdout.decode("utf-8", errors="replace")
    stderr = proc.stderr.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        log.info("git %s -> %s: %s", args[0], proc.returncode, stderr.strip())
        raise ToolInputError(_git_error_text(stderr, shown))
    return stdout


def _git_relpath(root: Path, target: Path) -> str:
    # git получает путь относительно cwd=REPO_ROOT; после "--" он не станет опцией
    return target.relative_to(root).as_posix() or "."


def git_log_impl(path: str, n: int = 5) -> str:
    if n < 1:
        raise ToolInputError("n должен быть >= 1")
    n = min(n, GIT_LOG_MAX)
    root, target = resolve_existing(path)
    # %B (сырое сообщение) + -z: коммиты разделены NUL, поля — \x1f.
    # Не %s: git склеивает в «subject» весь первый абзац, а нужна первая строка.
    args = ["log", f"-n{n}", "-z", "--no-color", "--format=%H%x1f%an%x1f%aI%x1f%B"]
    if target.is_file():
        args.append("--follow")  # история сквозь переименования
    out = run_git([*args, "--", _git_relpath(root, target)], root, path)

    rows = []
    for rec in filter(None, out.split("\0")):
        sha, author, date, message = (rec.lstrip("\n").split("\x1f", 3) + ["", "", ""])[:4]
        subject = message.strip().split("\n", 1)[0].strip()
        rows.append(f"{sha[:7]} | {author} | {date} | {subject}")
    if not rows:
        return f"нет коммитов для {path} (файл не отслеживается git?)"
    return cap_text("\n".join(rows))


def _parse_porcelain(out: str) -> dict[str, str]:
    """Разбор `git blame --porcelain` для одной строки.

    Формат: заголовок "<sha40> <orig> <final> <count>", затем строки
    "ключ значение" (author, author-time, author-tz, summary, filename, ...),
    затем содержимое строки с префиксом табуляции.
    """
    lines = out.splitlines()
    if not lines:
        raise ToolInputError("git blame вернул пустой ответ")
    header = lines[0].split()
    info: dict[str, str] = {"sha": header[0], "orig_line": header[1], "final_line": header[2]}
    for line in lines[1:]:
        if line.startswith("\t"):
            info["content"] = line[1:]
            break
        key, _, value = line.partition(" ")
        info[key] = value
    return info


def _iso_from_git(ts: str, tz: str) -> str:
    sign = -1 if tz.startswith("-") else 1
    offset = timedelta(hours=int(tz[1:3]), minutes=int(tz[3:5])) * sign
    return datetime.fromtimestamp(int(ts), timezone(offset)).isoformat()


def git_blame_impl(path: str, line: int) -> str:
    if line < 1:
        raise ToolInputError("line должен быть >= 1")
    root, target = resolve_existing(path)
    if not target.is_file():
        raise ToolInputError(f"это не файл: {path}")
    out = run_git(
        ["blame", "-L", f"{line},{line}", "--porcelain", "--", _git_relpath(root, target)],
        root, path,
    )
    b = _parse_porcelain(out)
    code = f" {line}|{b.get('content', '')}"
    if b["sha"] == _ZERO_SHA:
        return f"{path}:{line} | строка ещё не закоммичена\n{code}"
    date = _iso_from_git(b.get("author-time", "0"), b.get("author-tz", "+0000"))
    head = " | ".join(
        [f"{path}:{line}", b["sha"][:7], b.get("author", "?"), date, b.get("summary", "")]
    )
    # filename в porcelain отсчитывается от корня git-репозитория, а REPO_ROOT
    # может быть его подкаталогом (data/repo/flask/src/flask): сравниваем хвост
    rel = _git_relpath(root, target)
    orig_file = b.get("filename", rel)
    if orig_file != rel and not orig_file.endswith("/" + rel):
        head += f" | исходно: {orig_file}:{b['orig_line']}"  # файл переименовывали
    return f"{head}\n{code}"


# --- MCP ---------------------------------------------------------------------
mcp = FastMCP("hopiumcode")


def _arg(tool: str, name: str):
    """Описание аргумента для JSON Schema (попадает в inputSchema.properties)."""
    return Field(description=ARG_DESCRIPTIONS[tool][name])


def _errors_as_text(fn):
    """Ошибки инструмента возвращаются обычным текстом «Ошибка: …» (ТЗ: «понятным текстом»).

    Не через исключение: SDK может скрыть текст исключения от клиента
    и прислать только «Error executing tool …», и модель не поймёт, что не так.
    functools.wraps сохраняет сигнатуру — по ней SDK строит схему аргументов.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ToolInputError as exc:
            return f"Ошибка: {exc}"
        except Exception as exc:  # noqa: BLE001 — неожиданное: в лог и коротко модели
            log.exception("инструмент %s упал", fn.__name__)
            return f"Ошибка: внутренняя ошибка инструмента {fn.__name__}: {type(exc).__name__}"
    return wrapper


def _describe(fn):
    """Описание инструмента берётся из docstring — так работает и в mcp 1.x, и в 2.x."""
    fn.__doc__ = TOOL_DESCRIPTIONS[fn.__name__]
    return fn


@mcp.tool()
@_describe
@_errors_as_text
def search_code(
    query: Annotated[str, _arg("search_code", "query")],
    k: Annotated[int, _arg("search_code", "k")] = 5,
) -> str:
    return search_code_impl(query, k)


@mcp.tool()
@_describe
@_errors_as_text
def read_file(
    path: Annotated[str, _arg("read_file", "path")],
    start_line: Annotated[int | None, _arg("read_file", "start_line")] = None,
    end_line: Annotated[int | None, _arg("read_file", "end_line")] = None,
) -> str:
    return read_file_impl(path, start_line, end_line)


@mcp.tool()
@_describe
@_errors_as_text
def find_callers(
    symbol: Annotated[str, _arg("find_callers", "symbol")],
    depth: Annotated[int, _arg("find_callers", "depth")] = 1,
) -> str:
    return find_callers_impl(symbol, depth)


@mcp.tool()
@_describe
@_errors_as_text
def git_log(
    path: Annotated[str, _arg("git_log", "path")],
    n: Annotated[int, _arg("git_log", "n")] = 5,
) -> str:
    return git_log_impl(path, n)


@mcp.tool()
@_describe
@_errors_as_text
def git_blame(
    path: Annotated[str, _arg("git_blame", "path")],
    line: Annotated[int, _arg("git_blame", "line")],
) -> str:
    return git_blame_impl(path, line)


def main() -> None:
    logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        log.info("REPO_ROOT = %s", get_repo_root())
    except ReadFileError as exc:
        log.error("%s", exc)
        sys.exit(1)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
