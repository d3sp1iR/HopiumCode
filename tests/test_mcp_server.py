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
        return [t.name for t in (await session.list_tools()).tools]


def test_server_starts_over_stdio(tmp_path):
    names = asyncio.run(asyncio.wait_for(_list_tools_over_stdio(tmp_path), 60))
    assert isinstance(names, list)


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
