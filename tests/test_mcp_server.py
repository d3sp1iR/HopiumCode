"""MCP-сервер: запуск по stdio и инструменты как обычные функции. Без Ollama и клона Flask."""
import asyncio
import os
import sys

import pytest

from hopiumcode.mcp_server import server


async def _list_tools_over_stdio(root):
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(command=sys.executable, args=["-m", "hopiumcode.mcp_server.server"],
                                   env={"REPO_ROOT": str(root), "PYTHONPATH": os.pathsep.join(sys.path)})
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        return (await session.list_tools()).tools


TOOLS = {"read_file", "search_code", "find_callers", "git_log", "git_blame"}


def test_server_lists_five_tools_over_stdio(tmp_path):
    tools = asyncio.run(asyncio.wait_for(_list_tools_over_stdio(tmp_path), 60))
    assert {t.name for t in tools} == TOOLS
    for t in tools:
        assert t.description == server.DESCRIPTIONS[t.name]
        schema = t.input_schema
        assert schema["properties"] and all(p.get("description") for p in schema["properties"].values()), t.name


def test_descriptions_short_and_cover_when_and_example():
    import re
    assert set(server.DESCRIPTIONS) == TOOLS
    for name, text in server.DESCRIPTIONS.items():
        assert len(re.findall(r"[.!?](?:\s|$)", text)) <= 3, name
        assert f"{name}(" in text, f"{name}: нет примера вызова"
        assert "Результат" in text, f"{name}: нет формата результата"


def test_root_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("REPO_ROOT", str(tmp_path))
    assert server.get_root() == tmp_path.resolve()


def test_root_from_config(tmp_path, monkeypatch):
    monkeypatch.delenv("REPO_ROOT", raising=False)
    monkeypatch.setenv("CG_REPO_ROOT", str(tmp_path / "src"))
    assert server.get_root() == (tmp_path / "src").resolve()


# --------------------------------------------------------------------------- read_file

@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "app.py").write_text("".join(f"line {i}\n" for i in range(1, 501)), encoding="utf-8")
    (root / "pkg" / "mod.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (root / "notes.txt").write_text("text\n", encoding="utf-8")
    (root / "empty.py").write_text("", encoding="utf-8")
    (tmp_path / "secret.py").write_text("PASSWORD = 'root:x:0:0'\n", encoding="utf-8")
    return root


def test_read_default_first_200_lines(repo):
    out = server.read_file(repo, "app.py")
    lines = out.splitlines()
    assert lines[0] == "[app.py:1-200] из 500 строк"
    assert lines[1] == "   1| line 1" and lines[200] == " 200| line 200"
    assert "start_line=201" in lines[-1]


def test_read_range_and_nested_path(repo):
    out = server.read_file(repo, "pkg/mod.py", 2, 2)
    assert out == "[pkg/mod.py:2-2] из 2 строк\n   2|     return 1"


@pytest.mark.parametrize("path", ["../secret.py", "pkg/../../secret.py", "..\\secret.py", "~/secret.py"])
def test_parent_paths_blocked(repo, path):
    out = server.read_file(repo, path)
    assert out.startswith("Ошибка: путь вне репозитория") and "root:x" not in out


def test_etc_passwd_blocked(repo):
    for path in ["../etc/passwd", "../../../../../../etc/passwd", "/etc/passwd"]:
        out = server.read_file(repo, path)
        assert out.startswith("Ошибка: путь вне репозитория") and "root:" not in out


def test_absolute_path_inside_repo_also_blocked(repo):
    assert server.read_file(repo, str(repo / "app.py")).startswith("Ошибка: путь вне репозитория")


def test_symlink_outside_blocked(repo):
    (repo / "link.py").symlink_to(repo.parent / "secret.py")
    (repo / "linkdir").symlink_to(repo.parent, target_is_directory=True)
    for path in ["link.py", "linkdir/secret.py"]:
        out = server.read_file(repo, path)
        assert out.startswith("Ошибка: путь вне репозитория") and "root:x" not in out


def test_symlink_inside_allowed(repo):
    (repo / "alias.py").symlink_to(repo / "pkg" / "mod.py")
    assert "def f():" in server.read_file(repo, "alias.py")


@pytest.mark.parametrize("path, text", [
    ("missing.py", "файл не найден"), ("pkg", "каталог"), ("notes.txt", "только файлы .py"), ("", "не указан путь"),
])
def test_bad_targets_return_text(repo, path, text):
    out = server.read_file(repo, path)
    assert out.startswith("Ошибка:") and text in out


def test_start_after_end_and_past_eof(repo):
    assert "больше end_line" in server.read_file(repo, "app.py", 10, 5)
    assert "за концом файла" in server.read_file(repo, "app.py", 600)
    assert server.read_file(repo, "app.py", 495, 900).splitlines()[0] == "[app.py:495-500] из 500 строк"
    assert server.read_file(repo, "empty.py") == "[empty.py] пустой файл"


def test_max_400_lines(repo):
    out = server.read_file(repo, "app.py", 1, 500)
    assert out.splitlines()[0] == "[app.py:1-400] из 500 строк" and "не больше 400 строк" in out


def test_big_file_truncated_to_8000_chars(repo):
    (repo / "big.py").write_text("".join(f"x_{i} = '{'a' * 100}'\n" for i in range(300)), encoding="utf-8")
    out = server.read_file(repo, "big.py", 1, 300)
    assert len(out) <= server.MAX_CHARS and "обрезано" in out
    shown = int(out.splitlines()[0].split("-")[1].split("]")[0])
    assert f"start_line={shown + 1}" in out and out.splitlines()[-2].startswith(f"{shown:>4}|")


def test_header_is_citation_for_orchestrator(repo):
    from hopiumcode.orchestrator import parse_citations
    assert parse_citations(server.read_file(repo, "app.py", 10, 20))[0][:3] == ("app.py", 10, 20)


# --------------------------------------------------------------------------- git

def _git(cwd, *args, date=None):
    import subprocess
    env = {**os.environ, "GIT_AUTHOR_NAME": "Alice", "GIT_AUTHOR_EMAIL": "a@example.com",
           "GIT_COMMITTER_NAME": "Alice", "GIT_COMMITTER_EMAIL": "a@example.com"}
    if date:
        env |= {"GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
    subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True)


@pytest.fixture
def git_repo(tmp_path):
    """Git-репозиторий, корень инструментов в подкаталоге src/ (как data/repo/flask/src/flask)."""
    top = tmp_path / "proj"
    src = top / "src"
    src.mkdir(parents=True)
    _git(top, "init", "-q", "-b", "main")
    (src / "app.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    _git(top, "add", ".")
    _git(top, "commit", "-q", "-m", "добавлен app.py", date="2024-01-02T10:00:00+03:00")
    (src / "app.py").write_text("def a():\n    return 2\n\n\ndef b():\n    return 3\n", encoding="utf-8")
    _git(top, "commit", "-q", "-am", "изменён a, добавлен b\n\nподробности в теле", date="2024-02-03T11:30:00+00:00")
    (src / "other.py").write_text("x = 1\n", encoding="utf-8")
    _git(top, "add", ".")
    _git(top, "commit", "-q", "-m", "other", date="2024-03-04T12:00:00+00:00")
    return src


def test_git_log_format_newest_first(git_repo):
    out = server.git_log(git_repo, "app.py")
    rows = out.splitlines()[1:]
    assert len(rows) == 2                                       # коммит other.py не трогал app.py
    sha, author, date, subject = rows[0].split(" | ")
    assert len(sha) == 7 and author == "Alice"
    assert date == "2024-02-03T11:30:00+00:00" and subject == "изменён a, добавлен b"
    assert rows[1].endswith("2024-01-02T10:00:00+03:00 | добавлен app.py")


def test_git_log_limit(git_repo):
    assert len(server.git_log(git_repo, "app.py", n=1).splitlines()) == 2


def test_git_blame_line(git_repo):
    first = server.git_blame(git_repo, "app.py", 1)
    assert first.startswith("[app.py:1-1] коммит") and "добавлен app.py" in first and "Alice" in first
    assert "2024-01-02T10:00:00+03:00" in first and first.splitlines()[1] == "   1| def a():"
    second = server.git_blame(git_repo, "app.py", 2)
    assert "изменён a, добавлен b" in second and "return 2" in second


def test_git_blame_uncommitted(git_repo):
    (git_repo / "app.py").write_text("def a():\n    return 99\n", encoding="utf-8")
    assert "не закоммичена" in server.git_blame(git_repo, "app.py", 2)


@pytest.mark.parametrize("call, text", [
    (lambda r: server.git_blame(r, "app.py", 100), "вне диапазона"),
    (lambda r: server.git_blame(r, "app.py", 0), "вне диапазона"),
    (lambda r: server.git_blame(r, "missing.py", 1), "файл не найден"),
    (lambda r: server.git_log(r, "missing.py"), "файл не найден"),
    (lambda r: server.git_log(r, "../outside.py"), "путь вне репозитория"),
    (lambda r: server.git_blame(r, "/etc/passwd", 1), "путь вне репозитория"),
])
def test_git_errors_as_text(git_repo, call, text):
    out = call(git_repo)
    assert out.startswith("Ошибка:") and text in out


def test_git_not_a_repo(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    out = server.git_log(tmp_path, "a.py")
    assert out.startswith("Ошибка:") and "git-репозитории" in out


def test_git_timeout_and_missing_git(git_repo, monkeypatch):
    import subprocess

    def timeout(*a, **kw):
        raise subprocess.TimeoutExpired("git", server.GIT_TIMEOUT)
    monkeypatch.setattr(server.subprocess, "run", timeout)
    assert "не ответил за 10 c" in server.git_log(git_repo, "app.py")

    def missing(*a, **kw):
        raise FileNotFoundError("git")
    monkeypatch.setattr(server.subprocess, "run", missing)
    assert "git не установлен" in server.git_blame(git_repo, "app.py", 1)


def test_git_called_without_shell(git_repo, monkeypatch):
    import subprocess
    calls = []
    real = subprocess.run

    def spy(args, **kw):
        calls.append((args, kw))
        return real(args, **kw)
    monkeypatch.setattr(server.subprocess, "run", spy)
    server.git_log(git_repo, "app.py; rm -rf /")                # имя файла не исполняется
    server.git_log(git_repo, "app.py")
    args, kw = calls[-1]
    assert isinstance(args, list) and not kw.get("shell") and kw["timeout"] == 10 and kw["cwd"] == git_repo.resolve()


# --------------------------------------------------------------------------- search_code и find_callers

@pytest.fixture
def calls_repo(tmp_path):
    root = tmp_path / "calls"
    (root / "pkg").mkdir(parents=True)
    (root / "validation.py").write_text("def validate(x):\n    return x\n", encoding="utf-8")
    (root / "pkg" / "app.py").write_text(
        "from validation import validate\n\n\n"
        "class Service:\n"
        "    @staticmethod\n"
        "    def register(x):\n"
        "        # validate(x) в комментарии не считается\n"
        "        return validate(x)\n\n\n"
        "def main():\n"
        "    return Service.register(1)\n", encoding="utf-8")
    return root


def test_search_code_stub_is_marked(calls_repo, capsys):
    out = server.search_code(calls_repo, "где проверяется ввод", 2)
    assert out.startswith("⚠ ЗАГЛУШКА") and "участник 3" in out
    assert "⚠ ЗАГЛУШКА" in capsys.readouterr().err
    from hopiumcode.orchestrator import parse_citations
    assert parse_citations(out) == []                          # демо-ссылки не становятся источниками


def test_search_code_real_retrieve_format(calls_repo, monkeypatch):
    from hopiumcode.models import Chunk, Hit
    seen = []

    def retrieve(q, k, mode):
        seen.append((q, k, mode))
        c = Chunk("pkg/app.py::Service.register", "pkg/app.py", 5, 8, "method", "Service.register",
                  "    @staticmethod\n    def register(x):\n" + "        return validate(x)  " * 20, "python")
        return [Hit(c, 0.9, "vector+bm25")]

    monkeypatch.setattr("hopiumcode.retrieval.hybrid.retrieve", retrieve)
    out = server.search_code(calls_repo, "register", 50)
    row = out.splitlines()[1]
    ref, kind, name, text = row.split(" | ", 3)
    assert (ref, kind, name) == ("[pkg/app.py:5-8]", "method", "Service.register")
    assert len(text) <= 120 and "\n" not in text
    assert seen == [("register", server.MAX_K, "hybrid")]
    from hopiumcode.orchestrator import parse_citations
    assert parse_citations(out)[0][:3] == ("pkg/app.py", 5, 8)


def test_search_code_empty_query(calls_repo):
    assert server.search_code(calls_repo, "  ").startswith("Ошибка:")


def test_find_callers_text_fallback(calls_repo, capsys):
    out = server.find_callers(calls_repo, "validate")
    assert "граф кода не реализован" in out and "текстовым поиском" in capsys.readouterr().err
    rows = out.splitlines()[2:]
    assert rows == ["[pkg/app.py:5-8] | Service.register"]      # комментарий и import не считаются


def test_find_callers_depth_2(calls_repo):
    rows = server.find_callers(calls_repo, "validate", depth=2).splitlines()[2:]
    assert rows[0].startswith("[pkg/app.py:5-8] | Service.register (уровень 1")
    assert rows[1].startswith("[pkg/app.py:11-12] | main (уровень 2, вызывает register)")


def test_find_callers_qualified_name_and_not_found(calls_repo):
    assert "[pkg/app.py:11-12] | main" in server.find_callers(calls_repo, "Service.register")
    assert "не найдено" in server.find_callers(calls_repo, "nobody_calls_me")
    assert server.find_callers(calls_repo, "rm -rf /").startswith("Ошибка:")


def test_find_callers_uses_graph_when_ready(calls_repo, monkeypatch):
    monkeypatch.setattr(server, "_graph_callers",
                        lambda root, symbol, depth: ["pkg/app.py::Service.register", "unknown::thing"])
    out = server.find_callers(calls_repo, "validate")
    assert "граф кода" in out.splitlines()[0] and "⚠" not in out
    assert out.splitlines()[1:] == ["[pkg/app.py:5-8] | Service.register", "unknown::thing | (строки не найдены)"]


def test_git_root_relative_path_is_accepted(git_repo):
    # корень инструментов proj/src, а модель передала путь от корня git
    assert server.read_file(git_repo, "src/app.py").startswith("[app.py:1-")
    assert server.git_log(git_repo, "src/app.py").startswith("git log app.py")
    assert server.read_file(git_repo, "src/../../x.py").startswith("Ошибка: путь вне репозитория")
