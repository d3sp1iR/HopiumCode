"""Загрузка и разбиение на фрагменты (участник 2)."""
from pathlib import Path

from hopiumcode.graph.extract import extract
from hopiumcode.indexing.chunker import chunk_file, split_lines
from hopiumcode.indexing.loader import iter_files

SAMPLE = '''\
"""Модуль."""
import os

X = 1


@deco
def top(a):
    """doc"""
    return a


class A(B):
    """cls doc"""
    attr = 1

    @property
    def m(self):
        return 1

    async def n(self):
        def inner():
            pass
        return 2

    class In:
        def q(self):
            pass


if X:
    def cond():
        pass

Y = top(2)
'''


def write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def by_id(chunks):
    return {c.id: c for c in chunks}


def test_loader_filters(tmp_path):
    write(tmp_path, "a.py", "x = 1\n")
    write(tmp_path, "docs/guide.md", "# T\n")
    write(tmp_path, "__pycache__/a.py", "x = 1\n")
    write(tmp_path, ".git/hooks.py", "x = 1\n")
    write(tmp_path, "pkg.egg-info/x.py", "x = 1\n")
    write(tmp_path, "data.bin", "x")
    write(tmp_path, "huge.py", "x = 1\n" * 100_000)
    (tmp_path / "bin.py").write_bytes(b"\x00\x01\x02binary")
    (tmp_path / "bad.txt").write_bytes(b"\xff\xfe\xfa\xfb" * 50)
    found = sorted(p.relative_to(tmp_path).as_posix() for p in iter_files(tmp_path))
    assert found == ["a.py", "docs/guide.md"]


def test_loader_skips_symlink_outside(tmp_path):
    outside = tmp_path / "outside"
    repo = tmp_path / "repo"
    write(outside, "secret.py", "x = 1\n")
    write(repo, "ok.py", "x = 1\n")
    try:
        (repo / "link.py").symlink_to(outside / "secret.py")
    except OSError:
        return          # нет прав на симлинки (Windows)
    assert [p.name for p in iter_files(repo)] == ["ok.py"]


def test_python_chunks(tmp_path):
    path = write(tmp_path, "m.py", SAMPLE)
    chunks = by_id(chunk_file(path, tmp_path))
    assert set(chunks) == {"m.py::top", "m.py::A", "m.py::A.m", "m.py::A.n", "m.py::A.In",
                           "m.py::A.In.q", "m.py::cond", "m.py", "m.py~2", "m.py~3"}
    top = chunks["m.py::top"]
    assert (top.start_line, top.end_line, top.kind) == (7, 10, "function")
    assert top.text.startswith("@deco") and '"""doc"""' in top.text      # декоратор и докстринг
    assert chunks["m.py::A.m"].kind == "method" and chunks["m.py::A.m"].start_line == 17
    assert chunks["m.py::A.n"].end_line == 24 and "def inner" in chunks["m.py::A.n"].text
    cls = chunks["m.py::A"]                  # заголовок класса без методов
    assert "attr = 1" in cls.text and "def m" not in cls.text and cls.kind == "class"
    module = chunks["m.py"]                  # код модуля: по фрагменту на участок между определениями
    assert module.kind == "module" and module.name is None and "import os" in module.text
    modules = [c for c in chunks.values() if c.kind == "module"]
    assert all("def top" not in c.text for c in modules)
    assert any("Y = top(2)" in c.text for c in modules)
    assert all(c.lang == "python" and c.path == "m.py" for c in chunks.values())
    lines = SAMPLE.splitlines()              # диапазон строк не накрывает чужой код
    for c in chunks.values():
        assert "\n".join(lines[c.start_line - 1:c.end_line]) == c.text, c.id


def test_python_edge_cases(tmp_path, caplog=None):
    assert chunk_file(write(tmp_path, "empty.py", ""), tmp_path) == []
    assert chunk_file(write(tmp_path, "bad.py", "def (:\n"), tmp_path) == []      # синтаксическая ошибка
    only_def = chunk_file(write(tmp_path, "d.py", "def f():\n    pass\n"), tmp_path)
    assert [c.id for c in only_def] == ["d.py::f"]                                 # модуля нет, кода нет


def test_markdown_rst_txt(tmp_path):
    md = chunk_file(write(tmp_path, "d.md", "intro\n\n# One\ntext\n\n## Two\nmore\n"), tmp_path)
    assert [(c.id, c.start_line, c.end_line, c.name) for c in md] == [
        ("d.md::#0", 1, 2, None), ("d.md::#1", 3, 5, "One"), ("d.md::#2", 6, 7, "Two")]
    assert all(c.kind == "doc" and c.lang == "markdown" for c in md)
    rst = chunk_file(write(tmp_path, "d.rst", "Title\n=====\n\ntext\n\nSub\n---\nmore\n"), tmp_path)
    assert len(rst) == 2 and rst[0].lang == "rst" and rst[0].name == "Title"
    txt = chunk_file(write(tmp_path, "n.txt", "hello\nworld\n"), tmp_path)
    assert [(c.id, c.lang) for c in txt] == [("n.txt::#0", "text")]


def test_long_function_is_split_with_overlap(tmp_path):
    body = "\n".join(f"    value_{i} = compute(arg_{i}, other_{i})" + ("\n" if i % 7 == 6 else "")
                     for i in range(120))
    path = write(tmp_path, "big.py", f"def big():\n{body}\n")
    chunks = [c for c in chunk_file(path, tmp_path) if c.name == "big"]
    assert len(chunks) > 2
    assert chunks[0].id == "big.py::big" and chunks[1].id == "big.py::big~2"
    assert chunks[0].start_line == 1 and chunks[-1].end_line == len(path.read_text().splitlines())
    lines = path.read_text().splitlines()
    for c in chunks:
        assert c.text == "\n".join(lines[c.start_line - 1:c.end_line])      # номера строк честные
    for a, b in zip(chunks, chunks[1:]):
        assert b.start_line <= a.end_line + 1                                # нет дыр
        assert b.start_line > a.start_line                                   # есть движение вперёд


def test_split_lines_single_huge_line_makes_progress():
    parts = split_lines(["word " * 1000, "x", "y"], 1)
    assert parts and parts[-1][1] == 3


def test_chunk_ids_match_graph_nodes(tmp_path):
    write(tmp_path, "pkg/__init__.py", "from .a import f\n")
    write(tmp_path, "pkg/a.py", "class K:\n    def m(self):\n        return f()\n\n\ndef f():\n    return 1\n\n\nx = f()\n")
    write(tmp_path, "README.md", "# Title\nUse `f` here.\n\n## Sub\n`K.m`\n")
    write(tmp_path, "docs/index.rst", "Intro\n=====\n\n.. autofunction:: pkg.a.f\n")
    ids = {c.id for p in iter_files(tmp_path) for c in chunk_file(p, tmp_path)}
    nodes, _ = extract(tmp_path)
    assert {n.id for n in nodes} == ids        # обход графа находит все фрагменты


def test_redefined_names_get_unique_ids(tmp_path):
    src = ("class P:\n    @property\n    def x(self):\n        return 1\n\n    @x.setter\n"
           "    def x(self, v):\n        pass\n\n\nif FLAG:\n    def f():\n        return 1\nelse:\n    def f():\n        return 2\n")
    chunks = chunk_file(write(tmp_path, "p.py", src), tmp_path)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    assert {"p.py::P.x", "p.py::P.x@2", "p.py::f", "p.py::f@2"} <= set(ids)
