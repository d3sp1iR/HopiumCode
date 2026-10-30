"""Тесты инструментов git_log и git_blame MCP-сервера (участник 4).

Каждый тест получает свежий временный git-репозиторий с детерминированной
историей: авторы и даты коммитов заданы через переменные окружения git.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hopiumcode.mcp_server import server
from hopiumcode.mcp_server.server import (
    ToolInputError,
    _iso_from_git,
    _parse_porcelain,
    git_blame_impl,
    git_log_impl,
)

LOG_ROW = re.compile(
    r"^[0-9a-f]{7} \| [^|]+ \| \d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d \| .+$"
)

ALICE = ("Alice", "alice@example.com")
BOB = ("Bob", "bob@example.com")
CAROL = ("Carol", "carol@example.com")


def make_git(workdir: Path):
    """Возвращает функцию git(*args, author=..., date=...) для этого каталога."""

    def git(*args: str, author=ALICE, date="2024-01-01T00:00:00+00:00") -> str:
        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": author[0], "GIT_AUTHOR_EMAIL": author[1],
            "GIT_COMMITTER_NAME": author[0], "GIT_COMMITTER_EMAIL": author[1],
            "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date,
        }
        proc = subprocess.run(
            ["git", *args], cwd=workdir, env=env, capture_output=True, text=True, check=True
        )
        return proc.stdout

    return git


@pytest.fixture
def git_env(tmp_path, monkeypatch):
    if shutil.which("git") is None:
        pytest.skip("git не установлен")
    # изоляция от ~/.gitconfig (gpgsign, hooksPath и т. п.) и от родительских репозиториев
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    for var in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("CG_REPO_ROOT", raising=False)
    return tmp_path


@pytest.fixture
def repo(git_env, monkeypatch):
    tmp_path = git_env
    root = tmp_path / "proj"
    (root / "pkg").mkdir(parents=True)
    git = make_git(root)
    git("init", "-q", "-b", "main")

    app = root / "pkg" / "app.py"
    app.write_text("import os\n\ndef main():\n    return 1\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-q", "-m", "Add app\n\nДлинное описание коммита",
        author=ALICE, date="2024-01-10T10:00:00+03:00")

    app.write_text("import os\n\ndef main():\n    return 2\n", encoding="utf-8")
    git("commit", "-q", "-am", "Fix return value", author=BOB, date="2024-02-20T12:30:00+00:00")

    (root / "README.md").write_text("# proj\n", encoding="utf-8")
    git("add", "README.md")
    git("commit", "-q", "-m", "Add readme", author=CAROL, date="2024-03-05T09:15:00-05:00")

    app.write_text("import os\n\ndef main():\n    return 2\n# end\n", encoding="utf-8")
    git("commit", "-q", "-am", "Add footer\nвторая строка сообщения",
        author=BOB, date="2024-04-01T00:00:00+00:00")

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("TOP SECRET\n", encoding="utf-8")

    monkeypatch.setenv("REPO_ROOT", str(root))
    return SimpleNamespace(root=root, git=git, outside=outside, tmp=tmp_path)


def symlink_or_skip(link: Path, target: Path, is_dir: bool = False) -> None:
    try:
        os.symlink(target, link, target_is_directory=is_dir)
    except (OSError, NotImplementedError):
        pytest.skip("нет прав на создание символических ссылок")


# --- git_log -------------------------------------------------------------------
def test_git_log_format_and_order(repo):
    rows = git_log_impl("pkg/app.py").splitlines()
    assert len(rows) == 3
    assert all(LOG_ROW.match(r) for r in rows), rows
    head7 = repo.git("rev-parse", "HEAD").strip()[:7]
    assert rows[0] == f"{head7} | Bob | 2024-04-01T00:00:00+00:00 | Add footer"
    assert rows[1].endswith("| Bob | 2024-02-20T12:30:00+00:00 | Fix return value")
    assert rows[2].endswith("| Alice | 2024-01-10T10:00:00+03:00 | Add app")


def test_git_log_only_first_line_of_message(repo):
    out = git_log_impl("pkg/app.py")
    assert "вторая строка" not in out and "Длинное описание" not in out


def test_git_log_n(repo):
    assert len(git_log_impl("pkg/app.py", n=1).splitlines()) == 1
    assert len(git_log_impl(".", n=1000).splitlines()) == 4  # n ограничен сверху, не ошибка


def test_git_log_directory(repo):
    out = git_log_impl("pkg")
    assert "Add readme" not in out and "Add app" in out


def test_git_log_bad_n(repo):
    with pytest.raises(ToolInputError, match="n должен быть"):
        git_log_impl("pkg/app.py", n=0)


def test_git_log_untracked_file(repo):
    (repo.root / "new.py").write_text("x = 1\n", encoding="utf-8")
    assert "нет коммитов" in git_log_impl("new.py")


def test_git_log_follows_rename(repo):
    repo.git("mv", "pkg/app.py", "pkg/main.py")
    repo.git("commit", "-q", "-m", "Rename app", author=CAROL, date="2024-05-01T00:00:00+00:00")
    out = git_log_impl("pkg/main.py")
    assert "Rename app" in out and "Add app" in out


def test_git_log_path_that_looks_like_option(repo):
    # после "--" имя файла не может стать опцией git
    (repo.root / "-p").write_text("trap\n", encoding="utf-8")
    repo.git("add", "--", "-p")
    repo.git("commit", "-q", "-m", "Add dash file", author=CAROL, date="2024-06-01T00:00:00+00:00")
    out = git_log_impl("-p")
    assert out.splitlines()[0].endswith("| Add dash file")
    assert "diff --git" not in out


# --- git_blame -----------------------------------------------------------------
def test_git_blame_first_line(repo):
    out = git_blame_impl("pkg/app.py", 1)
    head, code = out.splitlines()
    sha7, author, date, summary = head.split(" | ")[1:5]
    assert re.fullmatch(r"[0-9a-f]{7}", sha7)
    assert (author, date, summary) == ("Alice", "2024-01-10T10:00:00+03:00", "Add app")
    assert head.startswith("pkg/app.py:1 | ")
    assert code == " 1|import os"


def test_git_blame_changed_lines(repo):
    assert "| Bob | 2024-02-20T12:30:00+00:00 | Fix return value" in git_blame_impl("pkg/app.py", 4)
    assert git_blame_impl("pkg/app.py", 4).splitlines()[1] == " 4|    return 2"
    assert "| Add footer" in git_blame_impl("pkg/app.py", 5)


def test_git_blame_line_out_of_range(repo):
    with pytest.raises(ToolInputError, match="строка вне диапазона: число строк в файле pkg/app.py — 5"):
        git_blame_impl("pkg/app.py", 99)


def test_git_blame_line_zero(repo):
    with pytest.raises(ToolInputError, match="line должен быть"):
        git_blame_impl("pkg/app.py", 0)


def test_git_blame_uncommitted_line(repo):
    with (repo.root / "pkg" / "app.py").open("a", encoding="utf-8") as f:
        f.write("print('wip')\n")
    out = git_blame_impl("pkg/app.py", 6)
    assert "не закоммичена" in out and out.endswith(" 6|print('wip')")


def test_git_blame_untracked_file(repo):
    (repo.root / "new.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ToolInputError, match="не отслеживается"):
        git_blame_impl("new.py", 1)


def test_git_blame_directory(repo):
    with pytest.raises(ToolInputError, match="это не файл"):
        git_blame_impl("pkg", 1)


def test_git_blame_after_rename_shows_origin(repo):
    repo.git("mv", "pkg/app.py", "pkg/main.py")
    repo.git("commit", "-q", "-m", "Rename app", author=CAROL, date="2024-05-01T00:00:00+00:00")
    out = git_blame_impl("pkg/main.py", 1)
    assert "| Alice |" in out and "исходно: pkg/app.py:1" in out


# --- REPO_ROOT как подкаталог git-репозитория (как data/repo/flask/src/flask) ---
def test_repo_root_is_subdirectory(repo, monkeypatch):
    monkeypatch.setenv("REPO_ROOT", str(repo.root / "pkg"))
    assert len(git_log_impl("app.py").splitlines()) == 3
    out = git_blame_impl("app.py", 1)
    assert out.startswith("app.py:1 | ") and "| Alice |" in out
    assert "исходно" not in out  # путь от корня git (pkg/app.py) — не переименование


# --- ошибки окружения и пути ----------------------------------------------------
@pytest.mark.parametrize("tool", [lambda p: git_log_impl(p), lambda p: git_blame_impl(p, 1)])
@pytest.mark.parametrize("path", ["../outside/secret.txt", "../../../../etc/passwd", "/etc/passwd"])
def test_path_outside_repo(repo, tool, path):
    with pytest.raises(ToolInputError, match="путь вне репозитория"):
        tool(path)


def test_absolute_path_outside(repo):
    with pytest.raises(ToolInputError, match="путь вне репозитория"):
        git_log_impl(str(repo.outside / "secret.txt"))


def test_symlink_outside(repo):
    symlink_or_skip(repo.root / "leak", repo.outside, is_dir=True)
    with pytest.raises(ToolInputError, match="путь вне репозитория"):
        git_blame_impl("leak/secret.txt", 1)


@pytest.mark.parametrize("tool", [lambda p: git_log_impl(p), lambda p: git_blame_impl(p, 1)])
def test_missing_file(repo, tool):
    with pytest.raises(ToolInputError, match="файл не найден"):
        tool("pkg/nope.py")


@pytest.mark.parametrize("tool", [lambda p: git_log_impl(p), lambda p: git_blame_impl(p, 1)])
def test_not_a_git_repo(git_env, monkeypatch, tool):
    plain = git_env / "plain"
    plain.mkdir()
    (plain / "a.py").write_text("a = 1\n", encoding="utf-8")
    monkeypatch.setenv("REPO_ROOT", str(plain))
    with pytest.raises(ToolInputError, match="не является git-репозиторием"):
        tool("a.py")


@pytest.mark.parametrize("tool", [lambda p: git_log_impl(p), lambda p: git_blame_impl(p, 1)])
def test_repo_without_commits(git_env, monkeypatch, tool):
    empty = git_env / "empty"
    empty.mkdir()
    make_git(empty)("init", "-q", "-b", "main")
    (empty / "a.py").write_text("a = 1\n", encoding="utf-8")
    monkeypatch.setenv("REPO_ROOT", str(empty))
    with pytest.raises(ToolInputError, match="нет коммитов"):
        tool("a.py")


# --- как вызывается subprocess ---------------------------------------------------
def test_subprocess_called_safely(repo, monkeypatch):
    calls = []
    real_run = subprocess.run

    def spy(cmd, *args, **kwargs):
        calls.append((cmd, kwargs))
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(server.subprocess, "run", spy)
    git_log_impl("pkg/app.py")
    git_blame_impl("pkg/app.py", 1)

    assert len(calls) == 2
    for cmd, kwargs in calls:
        assert isinstance(cmd, list) and cmd[0] == "git"
        assert not kwargs.get("shell")
        assert Path(kwargs["cwd"]) == repo.root.resolve()
        assert kwargs["timeout"] == 10
        assert cmd[-2] == "--"  # путь всегда после разделителя
    assert calls[1][0][-5:-2] == ["-L", "1,1", "--porcelain"]


def test_timeout(repo, monkeypatch):
    def hang(cmd, *args, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(server.subprocess, "run", hang)
    with pytest.raises(ToolInputError, match="не ответил за 10 секунд"):
        git_log_impl("pkg/app.py")


def test_git_not_installed(repo, monkeypatch):
    def missing(cmd, *args, **kwargs):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(server.subprocess, "run", missing)
    with pytest.raises(ToolInputError, match="git не установлен"):
        git_blame_impl("pkg/app.py", 1)


# --- разбор porcelain без git ----------------------------------------------------
def test_parse_porcelain():
    out = (
        "2afd7deedf62b211bdc8c6ec77430b64ee3f897a 3 7 1\n"
        "author Иван Петров\n"
        "author-mail <ivan@example.com>\n"
        "author-time 1700000000\n"
        "author-tz -0530\n"
        "summary Исправить | разделитель\n"
        "previous 1111111111111111111111111111111111111111 old.py\n"
        "filename old.py\n"
        "\t    return x  # таб внутри\tстроки\n"
    )
    b = _parse_porcelain(out)
    assert b["sha"].startswith("2afd7de")
    assert (b["orig_line"], b["final_line"]) == ("3", "7")
    assert b["author"] == "Иван Петров"
    assert b["summary"] == "Исправить | разделитель"
    assert b["filename"] == "old.py"
    assert b["content"] == "    return x  # таб внутри\tстроки"
    assert _iso_from_git(b["author-time"], b["author-tz"]) == "2023-11-14T16:43:20-05:30"


def test_path_from_git_root_is_accepted(repo, monkeypatch):
    # корень инструментов proj/pkg, а модель передала путь от корня git (pkg/app.py)
    monkeypatch.setenv("REPO_ROOT", str(repo.root / "pkg"))
    rows = git_log_impl("pkg/app.py").splitlines()
    assert len(rows) == 3 and rows[0].endswith("| Add footer")
    assert len(git_log_impl("app.py").splitlines()) == 3
    with pytest.raises(Exception, match="вне репозитория"):
        git_log_impl("pkg/../../outside/secret.txt")
