"""indexing.loader.iter_files на временной директории."""
from pathlib import Path

import pytest

from hopiumcode.indexing.loader import MAX_BYTES, iter_files

FLASK_SRC = Path(__file__).resolve().parents[1] / "data" / "repo" / "flask" / "src" / "flask"


def _make(root: Path, rel: str, data: bytes | str = "x = 1\n") -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
    return p


def rels(root: Path) -> list[str]:
    return [p.relative_to(root).as_posix() for p in iter_files(root)]


def test_only_py_sorted(tmp_path):
    for rel in ["b.py", "a.py", "pkg/z.py", "pkg/__init__.py", "README.md", "docs/x.rst", "notes.txt"]:
        _make(tmp_path, rel)
    assert rels(tmp_path) == ["a.py", "b.py", "pkg/__init__.py", "pkg/z.py"]


@pytest.mark.parametrize("skip", [".git", "__pycache__", ".venv", "node_modules"])
def test_skips_service_dirs(tmp_path, skip):
    _make(tmp_path, "ok.py")
    _make(tmp_path, f"{skip}/hidden.py")
    _make(tmp_path, f"pkg/{skip}/deep.py")
    assert rels(tmp_path) == ["ok.py"]


def test_skips_binary_and_large(tmp_path):
    _make(tmp_path, "ok.py")
    _make(tmp_path, "bin.py", b"\x00\x01\x02code")
    _make(tmp_path, "latin1.py", "x = 'caf\xe9'\n".encode("latin-1"))
    _make(tmp_path, "big.py", "x = 1\n" * (MAX_BYTES // 6 + 10))
    _make(tmp_path, "edge.py", "#" * MAX_BYTES)                    # ровно лимит ещё берём
    assert rels(tmp_path) == ["edge.py", "ok.py"]


def test_utf8_cyrillic_is_text(tmp_path):
    _make(tmp_path, "ru.py", "# комментарий " * 2000)               # граница 8 КБ внутри кириллицы
    assert rels(tmp_path) == ["ru.py"]


def test_stable_order(tmp_path):
    for rel in ["c/a.py", "b.py", "a/b/c.py", "a_b.py", "A.py"]:
        _make(tmp_path, rel)
    assert rels(tmp_path) == rels(tmp_path) == sorted(rels(tmp_path))


def test_flask_has_24_files():
    if not FLASK_SRC.is_dir():
        pytest.skip("data/repo/flask не склонирован (make fetch)")
    assert len(rels(FLASK_SRC)) == 24
