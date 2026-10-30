"""Все модули импортируются; заглушки честно сообщают, чья они зона."""
import importlib
import re

import pytest

MODULES = [
    "hopiumcode.config", "hopiumcode.models", "hopiumcode.interfaces", "hopiumcode.prompts",
    "hopiumcode.orchestrator", "hopiumcode.cli", "hopiumcode.mcp_client",
    "hopiumcode.indexing.loader", "hopiumcode.indexing.chunker", "hopiumcode.indexing.embed",
    "hopiumcode.indexing.vectorstore", "hopiumcode.indexing.build",
    "hopiumcode.retrieval.lexical", "hopiumcode.retrieval.hybrid",
    "hopiumcode.graph.extract", "hopiumcode.graph.store", "hopiumcode.graph.query",
    "hopiumcode.mcp_server.server",
]


@pytest.mark.parametrize("name", MODULES)
def test_importable(name):
    importlib.import_module(name)


def test_no_unimplemented_stubs_left():
    """Заглушки участников заменены реализациями: «raise NotImplementedError("участник N")» не осталось."""
    import pathlib

    import hopiumcode
    root = pathlib.Path(hopiumcode.__file__).parent
    left = [p.relative_to(root).as_posix() for p in root.rglob("*.py")
            if re.search(r'raise NotImplementedError\(\s*["\']участник', p.read_text(encoding="utf-8"))]
    assert left == []


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("CG_TOP_K", "5")
    from hopiumcode.config import Config
    assert Config.from_env().top_k == 5
