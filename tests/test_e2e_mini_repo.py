"""Сквозной путь на tests/fixtures/mini_repo: вопрос -> retrieve -> ответ (ScriptedChat) -> verify_citations.

Фрагменты берутся из indexing.chunker.chunk_file, а пока он не реализован (участник 2), из простого
разбиения через ast прямо в тесте. Поиск: пересечение слов вопроса и фрагмента. Без Ollama и клона Flask.
"""
import ast
import re
from pathlib import Path

import pytest

from hopiumcode.config import Config
from hopiumcode.models import Chunk, Hit
from hopiumcode.orchestrator import Agent, ModelReply, parse_citations, verify_citations
from hopiumcode.stubs import ScriptedChat, StubTools

MINI = Path(__file__).parent / "fixtures" / "mini_repo"
CFG = Config(top_k=4, max_steps=2, repo_name="mini_repo")
WORD = re.compile(r"\w+")


def _ast_chunks(path: Path) -> list[Chunk]:
    rel = path.relative_to(MINI).as_posix()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    lines = path.read_text(encoding="utf-8").splitlines()
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            text = "\n".join(lines[node.lineno - 1:node.end_lineno])
            out.append(Chunk(f"{rel}::{node.name}", rel, node.lineno, node.end_lineno, kind, node.name, text, "python"))
    return out


def mini_chunks() -> list[Chunk]:
    from hopiumcode.indexing.chunker import chunk_file
    chunks = []
    for path in sorted(MINI.glob("*.py")):
        try:
            chunks += chunk_file(path, MINI)
        except NotImplementedError:
            chunks += _ast_chunks(path)
    return chunks


CHUNKS = mini_chunks()


def mini_retrieve(question: str, k: int = 8, mode: str = "hybrid") -> list[Hit]:
    words = {w.lower() for w in WORD.findall(question)}
    scored = []
    for c in CHUNKS:
        overlap = len(words & {w.lower() for w in WORD.findall(c.text)})
        overlap += 2 * bool(c.name and c.name.lower() in words)     # совпадение с именем символа важнее
        if overlap:
            scored.append((overlap, c))
    if not scored:
        return []
    top = max(s for s, _ in scored)
    scored.sort(key=lambda x: (-x[0], x[1].id))
    return [Hit(c, s / top, "test-bm25") for s, c in scored[:k]]


def test_mini_repo_chunks_cover_all_files():
    assert {c.path for c in CHUNKS} == {"app.py", "cli.py", "storage.py", "validation.py"}


@pytest.mark.parametrize("question, expected_path", [
    ("Кто вызывает validate_email?", "validation.py"),
    ("Как save_user получает новый ключ next_id?", "storage.py"),
    ("Что делает register?", "app.py"),
])
def test_question_to_verified_citation(question, expected_path):
    retrieved: list[Hit] = []

    def retrieve(q, k, mode):
        retrieved[:] = mini_retrieve(q, k, mode)
        return retrieved

    ans = Agent(ScriptedChat(), retrieve, StubTools(), CFG).answer(question)
    assert retrieved and retrieved[0].chunk.path == expected_path
    assert ans.citations and not ans.bad_citations
    allowed = [(h.chunk.path, h.chunk.start_line, h.chunk.end_line) for h in retrieved]
    assert verify_citations(" ".join(ans.citations), allowed).ok
    for path, start, end, _ in parse_citations(" ".join(ans.citations)):
        n_lines = len((MINI / path).read_text(encoding="utf-8").splitlines())
        assert 1 <= start <= end <= n_lines                   # ссылка указывает на реальные строки


def test_caller_is_among_hits():
    hits = mini_retrieve("Кто вызывает validate_email?", 4)
    assert "app.py::register" in {h.chunk.id for h in hits}  # вызов внутри register тоже найден


def test_invented_citation_is_rejected():
    class Liar:
        def chat(self, messages, tools=None):
            return ModelReply("Ключ создаётся в [storage.py:100-120] и [nowhere.py:1-2].")

    ans = Agent(Liar(), mini_retrieve, StubTools(), CFG).answer("Как save_user получает ключ?")
    assert not ans.citations and len(ans.bad_citations) == 2
    assert "Непроверенные ссылки" in ans.text


def test_no_context_gives_honest_refusal():
    ans = Agent(ScriptedChat(), mini_retrieve, StubTools(), CFG).answer("Где настроен rate limiting?")
    assert not ans.citations and "ответа нет" in ans.text
