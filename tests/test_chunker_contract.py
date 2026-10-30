"""Контракты индексации и поиска (models.py, docs/contracts.md) на tests/fixtures/mini_repo.

Chunk: id по docs/contracts.md ("путь::Qualified.name", код модуля "путь", раздел текста "путь::#N",
продолжение "~N", повтор имени "@N"); путь относительно корня; текст равен строкам файла из диапазона.
Hit: score в 0..1. Пока реализация бросает NotImplementedError (участники 2 и 3), тест пропускается.
"""
import re
import typing
from pathlib import Path

import pytest

from hopiumcode.models import Chunk, ChunkKind, Hit
from hopiumcode.stubs import stub_retrieve

MINI = Path(__file__).parent / "fixtures" / "mini_repo"
ID_RE = re.compile(r"^(?P<path>[^:~@]+?)(?:::(?P<name>#\d+|[A-Za-z_][\w.]*))?(?:@\d+)?(?:~\d+)?$")
KINDS = set(typing.get_args(ChunkKind))


def _skip_if_stub(fn, *args, who: str):
    try:
        return fn(*args)
    except NotImplementedError as ex:
        pytest.skip(f"{who} ещё не реализован ({ex})")


def check_chunk(c: Chunk, root: Path) -> None:
    m = ID_RE.match(c.id)
    assert m, f"id {c.id!r} не по правилам docs/contracts.md"
    assert m["path"] == c.path, "путь в id и Chunk.path расходятся"
    assert not Path(c.path).is_absolute() and "\\" not in c.path, "путь должен быть относительным, через /"
    assert c.kind in KINDS and isinstance(c.text, str) and c.lang
    if m["name"] is None:
        assert c.kind == "module", "id без '::' бывает только у кода модуля"
    elif not m["name"].startswith("#"):
        assert c.name == m["name"], "qualified_name в id и Chunk.name расходятся"
    lines = (root / c.path).read_text(encoding="utf-8").splitlines()
    assert 1 <= c.start_line <= c.end_line <= len(lines)
    assert "\n".join(lines[c.start_line - 1:c.end_line]) == c.text, "текст не совпадает с диапазоном строк"
    assert c.ref == f"[{c.path}:{c.start_line}-{c.end_line}]"


def check_hits(hits: list[Hit]) -> None:
    assert isinstance(hits, list)
    for h in hits:
        assert isinstance(h, Hit) and isinstance(h.chunk, Chunk)
        assert 0.0 <= h.score <= 1.0, f"score {h.score} вне 0..1"
        assert isinstance(h.via, str) and h.via


def test_stub_hits_follow_contract():
    hits = stub_retrieve("вопрос", 3, "hybrid")
    check_hits(hits)
    for h in hits:
        assert ID_RE.match(h.chunk.id)


def test_iter_files_on_mini_repo():
    from hopiumcode.indexing.loader import iter_files
    files = list(_skip_if_stub(iter_files, MINI, who="indexing.loader.iter_files"))
    names = {Path(f).name for f in files}
    assert {"app.py", "cli.py", "storage.py", "validation.py"} <= names
    assert all("__pycache__" not in Path(f).parts for f in files)


def test_chunk_file_on_mini_repo():
    from hopiumcode.indexing.chunker import chunk_file
    chunks = _skip_if_stub(chunk_file, MINI / "storage.py", MINI, who="indexing.chunker.chunk_file")
    assert chunks
    for c in chunks:
        check_chunk(c, MINI)
        assert c.path == "storage.py" and c.lang == "python"
    names = {c.name for c in chunks}
    assert {"next_id", "save_user"} <= names, f"нет функций storage.py среди {names}"
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids)), "id фрагментов повторяются"


def test_chunk_file_kinds():
    from hopiumcode.indexing.chunker import chunk_file
    chunks = _skip_if_stub(chunk_file, MINI / "validation.py", MINI, who="indexing.chunker.chunk_file")
    by_name = {c.name: c for c in chunks}
    assert "InvalidEmail" in by_name and by_name["InvalidEmail"].kind == "class"
    assert by_name["validate_email"].kind == "function"


def test_fuse_scores_and_via():
    from hopiumcode.retrieval.lexical import fuse
    a = Chunk("a.py::f", "a.py", 1, 2, "function", "f", "def f(): ...", "python")
    b = Chunk("b.py::g", "b.py", 1, 2, "function", "g", "def g(): ...", "python")
    fused = _skip_if_stub(fuse, [[Hit(a, 0.9, "vector"), Hit(b, 0.5, "vector")], [Hit(a, 0.7, "bm25")]],
                          who="retrieval.lexical.fuse")
    check_hits(fused)
    assert fused[0].chunk.id == "a.py::f"                    # найден обоими списками -> первый
    assert "+" in fused[0].via                                # источники объединены через «+»
