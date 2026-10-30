"""Тесты инструмента read_file MCP-сервера (участник 4)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from hopiumcode.mcp_server import server
from hopiumcode.mcp_server.server import MAX_CHARS, MAX_LINES, ReadFileError, read_file_impl

ROW = re.compile(r"^ +(\d+)\|")  # " 12|текст"
SECRET = "TOP SECRET"


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "app.py").write_text(
        "import os\n\ndef main():\n    return 42\n", encoding="utf-8"
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text(SECRET + "\n", encoding="utf-8")
    # каталог-«соседка» с тем же префиксом: ловит наивный str.startswith
    (tmp_path / "repo_evil").mkdir()
    (tmp_path / "repo_evil" / "x.txt").write_text(SECRET + "\n", encoding="utf-8")

    monkeypatch.setenv("REPO_ROOT", str(root))
    monkeypatch.delenv("CG_REPO_ROOT", raising=False)
    return SimpleNamespace(root=root, outside=outside)


def symlink_or_skip(link: Path, target: Path, is_dir: bool = False) -> None:
    try:
        os.symlink(target, link, target_is_directory=is_dir)
    except (OSError, NotImplementedError):
        pytest.skip("нет прав на создание символических ссылок (Windows без Developer Mode)")


def numbers(text: str) -> list[int]:
    return [int(m.group(1)) for line in text.splitlines() if (m := ROW.match(line))]


def assert_outside(path: str) -> None:
    with pytest.raises(ReadFileError, match="путь вне репозитория") as exc:
        read_file_impl(path)
    assert SECRET not in str(exc.value)


# --- обычное чтение ----------------------------------------------------------
def test_reads_whole_file_with_numbers(repo):
    assert read_file_impl("pkg/app.py") == (
        " 1|import os\n 2|\n 3|def main():\n 4|    return 42"
    )


def test_reads_line_range(repo):
    assert read_file_impl("pkg/app.py", 3, 4) == " 3|def main():\n 4|    return 42"


def test_end_line_beyond_eof_is_clamped(repo):
    assert numbers(read_file_impl("pkg/app.py", 2, 100)) == [2, 3, 4]


@pytest.mark.parametrize(
    "start, end, msg",
    [(0, None, ">= 1"), (10, None, "больше числа строк"), (3, 2, "меньше start_line")],
)
def test_bad_ranges(repo, start, end, msg):
    with pytest.raises(ReadFileError, match=msg):
        read_file_impl("pkg/app.py", start, end)


# --- выход за пределы репозитория ---------------------------------------------
@pytest.mark.parametrize(
    "path",
    [
        "../etc/passwd",
        "../../../../../../../../etc/passwd",
        "../outside/secret.txt",
        "pkg/../../outside/secret.txt",
        "../repo_evil/x.txt",
    ],
)
def test_traversal_is_rejected(repo, path):
    assert_outside(path)


def test_absolute_path_outside_is_rejected(repo):
    assert_outside(str(repo.outside / "secret.txt"))
    assert_outside("/etc/passwd")


def test_absolute_path_inside_repo_is_rejected_too(repo):
    # ТЗ, раздел 10: абсолютные пути запрещены целиком, даже если ведут внутрь
    assert_outside(str(repo.root / "pkg" / "app.py"))


@pytest.mark.parametrize("path", ["pkg/../pkg/app.py", "pkg\\..\\pkg\\app.py"])
def test_dotdot_rejected_even_if_inside(repo, path):
    assert_outside(path)


@pytest.mark.parametrize("path", ["\\etc\\passwd", "C:\\Windows\\win.ini", "C:secret.txt"])
def test_windows_style_absolute_paths(repo, path):
    assert_outside(path)


# --- символические ссылки -------------------------------------------------------
def test_symlink_to_file_outside_is_rejected(repo):
    symlink_or_skip(repo.root / "leak.txt", repo.outside / "secret.txt")
    assert_outside("leak.txt")


def test_symlink_to_dir_outside_is_rejected(repo):
    symlink_or_skip(repo.root / "linkdir", repo.outside, is_dir=True)
    assert_outside("linkdir/secret.txt")


def test_symlink_inside_repo_is_allowed(repo):
    symlink_or_skip(repo.root / "alias.py", repo.root / "pkg" / "app.py")
    assert read_file_impl("alias.py", 1, 1) == " 1|import os"


# --- несуществующий файл и прочее -----------------------------------------------
def test_missing_file(repo):
    with pytest.raises(ReadFileError, match="файл не найден"):
        read_file_impl("pkg/nope.py")


def test_missing_file_outside_reports_outside_not_missing(repo):
    # снаружи нельзя проверять существование файлов
    assert_outside("../nope.txt")


def test_directory_is_not_a_file(repo):
    with pytest.raises(ReadFileError, match="не файл"):
        read_file_impl("pkg")


def test_binary_file_is_rejected(repo):
    (repo.root / "blob.bin").write_bytes(b"\x89PNG\x00\x01\x02")
    with pytest.raises(ReadFileError, match="бинарный"):
        read_file_impl("blob.bin")


def test_empty_file(repo):
    (repo.root / "empty.py").write_text("", encoding="utf-8")
    assert read_file_impl("empty.py") == "(пустой файл)"


def test_repo_root_not_set_and_no_default(repo, monkeypatch, tmp_path):
    monkeypatch.delenv("REPO_ROOT")
    monkeypatch.delenv("CG_REPO_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)                  # каталога по умолчанию из Config здесь нет
    with pytest.raises(ReadFileError, match="REPO_ROOT"):
        read_file_impl("pkg/app.py")


def test_repo_root_defaults_to_config(repo, monkeypatch, tmp_path):
    monkeypatch.delenv("REPO_ROOT")
    monkeypatch.delenv("CG_REPO_ROOT", raising=False)
    default = tmp_path / "data" / "repo" / "flask" / "src" / "flask"
    default.mkdir(parents=True)
    (default / "app.py").write_text("import os\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)                  # Config.repo_root относителен текущего каталога
    assert read_file_impl("app.py", 1, 1) == " 1|import os"


def test_cg_repo_root_fallback(repo, monkeypatch):
    monkeypatch.delenv("REPO_ROOT")
    monkeypatch.setenv("CG_REPO_ROOT", str(repo.root))
    assert read_file_impl("pkg/app.py", 1, 1) == " 1|import os"


# --- большие файлы ------------------------------------------------------------
@pytest.fixture
def many_lines(repo):
    (repo.root / "big.py").write_text(
        "".join(f"x{i}\n" for i in range(1, 10_001)), encoding="utf-8"
    )
    return "big.py"


def test_big_file_limited_to_400_lines(many_lines):
    out = read_file_impl(many_lines)
    assert numbers(out) == list(range(1, MAX_LINES + 1))
    assert "обрезано" in out
    assert "start_line=401" in out
    assert len(out) <= MAX_CHARS


def test_big_file_continuation(many_lines):
    out = read_file_impl(many_lines, start_line=401)
    assert numbers(out)[0] == 401
    assert out.splitlines()[0] == " 401|x401"


def test_long_lines_limited_to_8000_chars(repo):
    (repo.root / "wide.py").write_text(("y" * 120 + "\n") * 300, encoding="utf-8")
    out = read_file_impl("wide.py")
    assert len(out) <= MAX_CHARS
    assert "обрезано" in out
    nums = numbers(out)
    assert 0 < len(nums) < 300 and nums == list(range(1, len(nums) + 1))
    # строки не режутся посередине
    for line in out.splitlines():
        if ROW.match(line):
            assert line.endswith("y" * 120)


def test_single_huge_line(repo):
    (repo.root / "min.js").write_text("z" * 50_000, encoding="utf-8")
    out = read_file_impl("min.js")
    assert len(out) <= MAX_CHARS
    assert "обрезано" in out and "не полностью" in out


def test_file_over_size_limit_is_rejected(repo, monkeypatch):
    monkeypatch.setattr(server, "MAX_FILE_BYTES", 1000)
    (repo.root / "huge.py").write_text("a\n" * 1000, encoding="utf-8")
    with pytest.raises(ReadFileError, match="слишком большой"):
        read_file_impl("huge.py")


# --- обёртка инструмента: ошибки текстом ----------------------------------------
def test_tool_returns_errors_as_text(repo):
    # сам инструмент (то, что зовёт MCP) не бросает, а отвечает «Ошибка: …»
    assert server.read_file("../outside/secret.txt").startswith("Ошибка: путь вне репозитория")
    assert server.read_file("pkg/nope.py") == "Ошибка: файл не найден: pkg/nope.py"
    assert server.read_file("pkg/app.py", 1, 1) == " 1|import os"


def test_tool_hides_unexpected_exception_details(repo, monkeypatch):
    def boom(*args):
        raise KeyError("секрет")

    monkeypatch.setattr(server, "read_file_impl", boom)
    out = server.read_file("pkg/app.py")
    assert out == "Ошибка: внутренняя ошибка инструмента read_file: KeyError"
