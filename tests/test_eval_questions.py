"""eval/questions.yaml: формат корректен, все ссылки указывают на реальный код data/repo/flask."""
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from eval.run_eval import (
    QUESTION_TYPES,
    QUESTIONS_PATH,
    load_questions,
    normalize_question,
    parse_range,
)

REPO = Path(__file__).resolve().parents[1] / "data" / "repo" / "flask"
QUESTIONS = load_questions()
IDS = [q["id"] for q in QUESTIONS]


@pytest.fixture
def repo() -> Path:
    if not (REPO / "src" / "flask").is_dir():
        pytest.skip("data/repo/flask не склонирован")
    return REPO


def test_yaml_parses_to_list():
    raw = yaml.safe_load(QUESTIONS_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, list) and raw


def test_ids_unique():
    assert len(IDS) == len(set(IDS))


@pytest.mark.parametrize("q", QUESTIONS, ids=IDS)
def test_required_fields(q):
    assert q["question"].strip()
    assert q["type"] in QUESTION_TYPES
    assert q["reference_answer"].strip()
    if q["type"] == "trap":
        assert not q["expected_files"] and not q["expected_ranges"]
    else:
        assert q["expected_files"] and q["expected_ranges"]


@pytest.mark.parametrize("q", QUESTIONS, ids=IDS)
def test_ranges_within_expected_files(q):
    for ref in q["expected_ranges"]:
        path, start, end = parse_range(ref)
        assert path in q["expected_files"], ref
        assert 1 <= start <= end, ref


def test_legacy_fields_still_read():
    q = normalize_question({"id": "x", "category": "graph", "question": "?",
                            "gold_files": ["a.py"], "gold_answer": "ответ"})
    assert q["type"] == "graph"
    assert q["expected_files"] == ["a.py"]
    assert q["reference_answer"] == "ответ"
    assert q["expected_ranges"] == []


@pytest.mark.parametrize("q", QUESTIONS, ids=IDS)
def test_expected_files_exist(q, repo):
    for f in q["expected_files"]:
        assert (repo / f).is_file(), f


@pytest.mark.parametrize("q", QUESTIONS, ids=IDS)
def test_ranges_fit_file_length(q, repo):
    for ref in q["expected_ranges"]:
        path, _, end = parse_range(ref)
        n_lines = len((repo / path).read_text(encoding="utf-8").splitlines())
        assert end <= n_lines, f"{ref}: в файле {n_lines} строк"


@pytest.mark.parametrize("q", QUESTIONS, ids=IDS)
def test_symbols_defined_in_expected_files(q, repo):
    sources = "\n".join((repo / f).read_text(encoding="utf-8") for f in q["expected_files"])
    for sym in q["expected_symbols"]:
        name = re.escape(sym.split(".")[-1])
        pattern = rf"^\s*(?:def|class)\s+{name}\b|^\s*{name}\s*[:=]"
        assert re.search(pattern, sources, re.MULTILINE), sym


@pytest.mark.parametrize("q", [q for q in QUESTIONS if q["expected_commits"]],
                         ids=lambda q: q["id"])
def test_commits_exist(q, repo):
    if not (repo / ".git").exists():
        pytest.skip("data/repo/flask без истории git")
    for sha in q["expected_commits"]:
        cmd = ["git", "-C", str(repo), "cat-file", "-e", f"{sha}^{{commit}}"]
        assert subprocess.run(cmd, check=False).returncode == 0, sha
