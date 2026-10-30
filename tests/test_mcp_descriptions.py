"""Описания инструментов: полнота и соответствие сигнатурам."""

import inspect
import re

import pytest

from hopiumcode.mcp_server import server
from hopiumcode.mcp_server.descriptions import ARG_DESCRIPTIONS, TOOL_DESCRIPTIONS

TOOLS = ["search_code", "read_file", "find_callers", "git_log", "git_blame"]


def sentences(text: str) -> int:
    # точка/«!»/«?» в конце предложения; точки внутри кавычек и путей ("app.py") не считаются
    without_quotes = re.sub(r'"[^"]*"|«[^»]*»', "Q", text)
    return len(re.findall(r"[.!?](?=\s|$)", without_quotes))


def test_all_tools_described():
    assert sorted(TOOL_DESCRIPTIONS) == sorted(TOOLS)
    assert sorted(ARG_DESCRIPTIONS) == sorted(TOOLS)


@pytest.mark.parametrize("name", TOOLS)
def test_description_shape(name):
    text = TOOL_DESCRIPTIONS[name]
    assert sentences(text) <= 3, text
    assert "используй" in text and "НЕ используй" in text
    assert f"Пример: {name}(" in text
    assert "Результат" in text


@pytest.mark.parametrize("name", TOOLS)
def test_args_match_signature(name):
    fn = getattr(server, name)
    fn = getattr(fn, "fn", fn)  # на случай, если декоратор вернёт обёртку
    assert list(inspect.signature(fn).parameters) == list(ARG_DESCRIPTIONS[name])
    assert fn.__doc__ == TOOL_DESCRIPTIONS[name]
