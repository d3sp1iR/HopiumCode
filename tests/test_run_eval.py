"""eval/run_eval.py на синтетических данных: recall@k, сводки, таблицы, разбор ответа судьи."""
import pytest

from eval.run_eval import (
    UNAVAILABLE,
    format_table,
    judge,
    markdown_table,
    mode_total_table,
    mode_type_table,
    parse_judge,
    recall_at_k,
    summarize,
    unavailable_rows,
    weakest,
)

EXP = ["src/flask/app.py", "src/flask/ctx.py"]


@pytest.mark.parametrize("ranked, k, expected", [
    (["src/flask/app.py", "src/flask/ctx.py"], 5, 1.0),
    (["a.py", "b.py", "c.py", "d.py", "e.py", "src/flask/app.py"], 5, 0.0),  # app.py на 6-м месте
    (["a.py", "b.py", "c.py", "d.py", "e.py", "src/flask/app.py"], 8, 0.5),
    (["src/flask/app.py"] * 5, 5, 0.5),                                      # повторы файла не считаются дважды
    ([], 8, 0.0),
])
def test_recall_at_k(ranked, k, expected):
    assert recall_at_k(EXP, ranked, k) == expected


def test_recall_without_expected_files_is_none():
    assert recall_at_k([], ["src/flask/app.py"], 5) is None


def _row(qid, typ, mode="hybrid", **kw):
    base = {"id": qid, "type": typ, "mode": mode, "question": f"вопрос {qid}", "recall@5": 0.5, "recall@8": 1.0,
            "hit_files": 1.0, "citations_verified": 1.0, "answered": True, "latency_s": 2.0, "steps": 1,
            "tool_calls": 0, "found_files": [], "answer": "ответ"}
    return {**base, **kw}


def test_summarize_by_type_and_overall():
    rows = [_row("q1", "semantic", **{"recall@5": 1.0}), _row("q2", "semantic", **{"recall@5": 0.0}),
            _row("q3", "trap", **{"recall@5": None, "recall@8": None, "answered": False})]
    s = summarize(rows)
    assert s["by_type"]["semantic"]["recall@5"] == 0.5
    assert s["by_type"]["trap"]["recall@5"] is None           # у ловушек нет эталонных файлов
    assert s["overall"]["n"] == 3 and s["overall"]["answered"] == round(2 / 3, 3)


def test_unavailable_mode_is_marked_not_averaged():
    qs = [{"id": "q1", "type": "semantic", "question": "?"}]
    summaries = {"vector": summarize(unavailable_rows(qs, "vector")), "hybrid": summarize([_row("q1", "semantic")])}
    statuses = {"vector": UNAVAILABLE, "hybrid": "есть"}
    rows = mode_type_table(summaries, statuses)
    assert rows[0][:4] == ["vector", "semantic", "1", UNAVAILABLE]
    assert rows[1][:5] == ["hybrid", "semantic", "1", "0.50", "1.00"]
    totals = mode_total_table(summaries, statuses)
    assert totals[0][1] == UNAVAILABLE and totals[0][3] == "-"


def test_judge_column_only_when_enabled():
    summaries = {"hybrid": summarize([_row("q1", "semantic", judge=2)])}
    assert len(mode_type_table(summaries, {}, with_judge=True)[0]) == len(mode_type_table(summaries, {})[0]) + 1


def test_format_table_aligns_columns():
    text = format_table(["режим", "тип", "n"], [["vector", "graph", "10"], ["hybrid", "semantic", "2"]])
    lines = text.splitlines()
    assert len({len(line) for line in lines}) == 1             # все строки одной ширины
    assert lines[1].startswith("------")
    assert lines[3].endswith(" 2") and lines[2].endswith("10")  # числа выровнены вправо


def test_markdown_table_escapes_pipes():
    md = markdown_table(["a", "b"], [["x|y", "1"]]).splitlines()
    assert md[0] == "| a | b |" and md[1] == "|---|---|"
    assert md[2] == "| x\\|y | 1 |"


def test_weakest_puts_unanswered_and_low_recall_first():
    rows = [_row("good", "semantic"), _row("low", "semantic", **{"recall@8": 0.0}),
            _row("fail", "semantic", answered=False), _row("na", "semantic", unavailable=True)]
    assert [r["id"] for r in weakest(rows, 3)] == ["fail", "low", "good"]


@pytest.mark.parametrize("text, score", [
    ('{"score": 2, "reason": "совпадает"}', 2),
    ('Думаю...\n```json\n{"score": "1", "reason": "неполно"}\n```', 1),
    ('{"score": 0, "reason": "выдумка"}', 0),
])
def test_parse_judge_valid(text, score):
    assert parse_judge(text)["score"] == score


@pytest.mark.parametrize("text", ["без json", '{"score": 5}', '{"score": true}', "{score: 2"])
def test_parse_judge_invalid_gives_none_and_error(text):
    res = parse_judge(text)
    assert res["score"] is None and res["error"]


def test_judge_survives_model_error():
    def broken(messages):
        raise ConnectionError("нет Ollama")
    res = judge({"question": "?", "reference_answer": "эталон"}, "ответ", broken)
    assert res["score"] is None and "нет Ollama" in res["error"]


def test_judge_sends_prompt_reference_and_answer():
    seen = []
    judge({"question": "Где?", "reference_answer": "ЭТАЛОН"}, "ОТВЕТ",
          lambda m: seen.append(m) or '{"score": 1, "reason": "ok"}')
    system, user = seen[0]
    assert "Опирайся только на эталон" in system["content"]
    assert "ЭТАЛОН" in user["content"] and "ОТВЕТ" in user["content"]


def test_history_answered_by_expected_commit():
    from eval.run_eval import score
    from hopiumcode.models import Answer
    q = {"type": "history", "expected_files": ["src/flask/ctx.py"], "expected_commits": ["c17f3793"]}
    ok = Answer(text="Последний коммит c17f379 от David Lord", used_tools=["git_log"])
    assert score(q, ok, [], str)["answered"] is True
    assert score(q, Answer(text="Не знаю"), [], str)["answered"] is False
