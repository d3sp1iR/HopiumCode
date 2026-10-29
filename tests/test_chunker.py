"""indexing.chunker: ast-фрагменты, номера строк, разбиение длинных фрагментов, уникальные id."""
import itertools
import logging
import textwrap
from pathlib import Path

import pytest

from hopiumcode.indexing.chunker import (
    MAX_TOKENS,
    OVERLAP,
    chunk_file,
    chunk_source,
    estimate_tokens,
)
from hopiumcode.indexing.loader import iter_files

FLASK_SRC = Path(__file__).resolve().parents[1] / "data" / "repo" / "flask" / "src" / "flask"

SAMPLE = textwrap.dedent('''\
    """Модуль-пример."""
    import functools

    CONST = 1


    def deco(f):
        return f


    @deco
    @functools.lru_cache
    def cached(x):
        """Кэшируемая."""
        def inner():
            return x
        return inner()


    class Outer:
        """Внешний класс."""
        attr = 1

        class Inner:
            def method(self):
                return 2

        @property
        def prop(self):
            return self.attr

        async def fetch(self):
            return await something()


    async def run():
        return 1


    if __name__ == "__main__":
        run()
''')


def by_name(chunks):
    return {c.name: c for c in chunks if c.name}


def lines_of(chunk, source):
    return "\n".join(source.splitlines()[chunk.start_line - 1:chunk.end_line])


def test_kinds_and_qualified_names():
    chunks = chunk_source(SAMPLE, "pkg/sample.py")
    names = by_name(chunks)
    assert {n: c.kind for n, c in names.items()} == {
        "deco": "function", "cached": "function", "Outer": "class", "Outer.Inner": "class",
        "Outer.Inner.method": "method", "Outer.prop": "method", "Outer.fetch": "method", "run": "function"}
    assert "inner" not in names                                  # вложенная функция остаётся в тексте родителя
    assert "def inner()" in names["cached"].text


def test_decorators_and_docstring_in_range():
    chunks = chunk_source(SAMPLE, "sample.py")
    cached = by_name(chunks)["cached"]
    assert cached.text.startswith("@deco\n@functools.lru_cache\ndef cached(x):")
    assert '"""Кэшируемая."""' in cached.text
    assert lines_of(cached, SAMPLE) == cached.text
    prop = by_name(chunks)["Outer.prop"]
    assert prop.text.lstrip().startswith("@property") and lines_of(prop, SAMPLE) == prop.text


def test_async_def():
    names = by_name(chunk_source(SAMPLE, "sample.py"))
    assert names["run"].text.startswith("async def run")
    assert names["Outer.fetch"].text.lstrip().startswith("async def fetch")


def test_class_text_excludes_methods_and_nested_classes():
    outer = by_name(chunk_source(SAMPLE, "sample.py"))["Outer"]
    assert '"""Внешний класс."""' in outer.text and "attr = 1" in outer.text
    assert "def prop" not in outer.text and "class Inner" not in outer.text


def test_module_chunks_without_definitions():
    chunks = chunk_source(SAMPLE, "sample.py")
    mods = [c for c in chunks if c.kind == "module"]
    assert [(m.id, m.start_line, m.end_line) for m in mods] == [("sample.py::#1", 1, 4), ("sample.py::#2", 40, 41)]
    assert all(m.name is None for m in mods)
    assert "import functools" in mods[0].text and 'if __name__ == "__main__":' in mods[1].text
    assert all("def " not in m.text and "class " not in m.text for m in mods)


def test_every_chunk_is_contiguous_slice_of_file():
    # диапазон строк любого фрагмента не накрывает чужой код: текст равен строкам файла
    for c in chunk_source(SAMPLE, "sample.py"):
        assert lines_of(c, SAMPLE) == c.text, c.id


def test_module_code_between_definitions_is_separate():
    src = "import os\n\n\ndef f():\n    return 1\n\n\nX = f()\n\n\ndef g():\n    return 2\n\n\nY = g()\n"
    mods = [(c.start_line, c.end_line) for c in chunk_source(src, "m.py") if c.kind == "module"]
    assert mods == [(1, 1), (8, 8), (15, 15)]


def test_ids_unique_and_formatted():
    src = textwrap.dedent('''\
        from typing import overload

        @overload
        def f(x: int) -> int: ...
        @overload
        def f(x: str) -> str: ...
        def f(x):
            return x

        class A:
            @property
            def v(self):
                return 1
            @v.setter
            def v(self, val):
                pass
    ''')
    chunks = chunk_source(src, "dup.py")
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    assert "dup.py::f" in ids and "dup.py::A.v" in ids
    assert sum(1 for c in chunks if c.name == "f") == 3       # имя сохраняется, id через #N
    assert all(c.id.startswith("dup.py::") for c in chunks)


def test_long_function_split_with_overlap():
    body = "\n\n".join(f"    step_{i} = compute(alpha, beta, gamma, delta, epsilon, zeta, eta, theta)\n"
                       f"    check(step_{i}, alpha, beta, gamma, delta, epsilon)" for i in range(40))
    src = f"def long():\n{body}\n    return 1\n"
    chunks = [c for c in chunk_source(src, "long.py") if c.name == "long"]
    assert len(chunks) > 1
    assert chunks[0].id == "long.py::long"                    # первая часть сохраняет имя
    for c in chunks[1:]:
        assert c.id.startswith("long.py::#")
    for c in chunks:
        assert lines_of(c, src) == c.text                     # текст части совпадает с её строками в файле
        assert estimate_tokens(c.text) <= MAX_TOKENS * 1.2
    src_lines = src.splitlines()
    for prev, nxt in itertools.pairwise(chunks):
        assert nxt.start_line <= prev.end_line                 # части перекрываются
        tail = [n for n in range(prev.start_line, prev.end_line + 1) if src_lines[n - 1].strip()][-OVERLAP:]
        assert nxt.start_line == tail[0]
    covered = set()
    for c in chunks:
        covered |= set(range(c.start_line, c.end_line + 1))
    assert covered == set(range(1, len(src_lines) + 1))      # ничего не потеряно


def test_line_numbers_match_text_for_all_contiguous_chunks():
    for c in chunk_source(SAMPLE, "sample.py"):
        if c.kind in ("function", "method"):
            assert lines_of(c, SAMPLE) == c.text


def test_empty_file(tmp_path):
    (tmp_path / "empty.py").write_text("")
    (tmp_path / "blank.py").write_text("\n\n   \n")
    assert chunk_file(tmp_path / "empty.py", tmp_path) == []
    assert chunk_file(tmp_path / "blank.py", tmp_path) == []


def test_syntax_error_skipped_with_warning(tmp_path, caplog):
    (tmp_path / "bad.py").write_text("def broken(:\n    pass\n")
    with caplog.at_level(logging.WARNING, logger="hopiumcode"):
        assert chunk_file(tmp_path / "bad.py", tmp_path) == []
    assert "bad.py" in caplog.text


def test_path_relative_with_slashes(tmp_path):
    (tmp_path / "pkg" / "sub").mkdir(parents=True)
    (tmp_path / "pkg" / "sub" / "m.py").write_text("def f():\n    return 1\n")
    chunks = chunk_file(tmp_path / "pkg" / "sub" / "m.py", tmp_path)
    assert [c.id for c in chunks] == ["pkg/sub/m.py::f"]
    c = chunks[0]
    assert (c.path, c.start_line, c.end_line, c.lang) == ("pkg/sub/m.py", 1, 2, "python")


def test_flask_files_and_chunks():
    if not FLASK_SRC.is_dir():
        pytest.skip("data/repo/flask не склонирован (make fetch)")
    files = list(iter_files(FLASK_SRC))
    chunks = [c for f in files for c in chunk_file(f, FLASK_SRC)]
    assert len(files) == 24
    assert len({c.id for c in chunks}) == len(chunks)
    names = {c.name for c in chunks}
    assert {"Flask.full_dispatch_request", "RequestContext.push", "SecureCookieSessionInterface.open_session",
            "Scaffold.route", "url_for"} <= names
    paths = {c.path for c in chunks}
    assert "app.py" in paths and "sansio/scaffold.py" in paths and "json/tag.py" in paths
    sources = {p: (FLASK_SRC / p).read_text(encoding="utf-8") for p in paths}
    for c in chunks:
        assert lines_of(c, sources[c.path]) == c.text, c.id    # непрерывный кусок файла
    assert max(c.end_line - c.start_line for c in chunks if c.kind == "module") < 100
