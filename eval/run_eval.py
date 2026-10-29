"""Сравнение режимов retrieve: vector | vector+bm25 | hybrid. Владелец: участник 5.

python -m eval.run_eval [--mode hybrid] [--fake] [--only q01,q05] [--out-dir eval/results]

Метрики на вопрос:
  hit_files           доля expected_files среди найденных фрагментов и подтверждённых ссылок ответа
  citations_verified  подтверждено ссылок / всего ссылок в ответе (None, если ссылок нет)
  answered            trap: агент сказал, что не нашёл, и не дал ни одной ссылки;
                      остальные типы: есть хотя бы одна подтверждённая ссылка
  latency_s, tool_calls
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = Path(__file__).with_name("questions.yaml")
RESULTS_DIR = Path(__file__).with_name("results")
TARGET_ROOT = ROOT / "data" / "repo" / "flask"  # от него считаются пути в questions.yaml
MODES = ("vector", "vector+bm25", "hybrid")
_REFUSAL_RE = re.compile(
    r"не\s+(?:на[шй]|нашл|удалось\s+найти|найден)|ответа\s+нет|нет\s+ответа|отсутству|нет\s+(?:в\s+(?:коде|контексте)|информации)",
    re.IGNORECASE)
QUESTION_TYPES = {"semantic", "graph", "history", "overview", "trap", "explain"}

# старое поле -> новое
_LEGACY_FIELDS = {"category": "type", "gold_files": "expected_files", "gold_answer": "reference_answer"}


def normalize_question(raw: dict) -> dict:
    """Приводит вопрос к новому формату; старые поля (category, gold_*) остаются читаемыми."""
    q = dict(raw)
    for old, new in _LEGACY_FIELDS.items():
        if old in q and new not in q:
            q[new] = q[old]
    q.setdefault("type", "semantic")
    for key in ("expected_files", "expected_symbols", "expected_ranges", "expected_commits"):
        q[key] = list(q.get(key) or [])
    q.setdefault("reference_answer", "")
    q.setdefault("notes", "")
    return q


def parse_range(ref: str) -> tuple[str, int, int]:
    """'путь:начало-конец' -> (путь, начало, конец)."""
    path, _, lines = ref.rpartition(":")
    start, _, end = lines.partition("-")
    return path, int(start), int(end or start)


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return [normalize_question(q) for q in data]


def path_mapper(repo_root: Path, target_root: Path = TARGET_ROOT) -> Callable[[str], str]:
    """Переводит путь фрагмента (относительно Config.repo_root) в путь относительно target_root."""
    root, target = repo_root.resolve(), target_root.resolve()
    if root.is_relative_to(target):
        prefix = root.relative_to(target).as_posix()  # "src/flask" или "."
        if prefix == ".":
            return lambda p: p
        return lambda p: p if p.startswith(prefix + "/") else f"{prefix}/{p}"
    if target.is_relative_to(root):
        strip = target.relative_to(root).as_posix() + "/"
        return lambda p: p.removeprefix(strip)
    return lambda p: p


def is_refusal(text: str) -> bool:
    return bool(_REFUSAL_RE.search(text))


def score(q: dict, answer, found_paths: set[str], to_target: Callable[[str], str]) -> dict:
    """Метрики одного ответа (Answer из models.py) по эталону вопроса."""
    from hopiumcode.orchestrator import parse_citations

    cited = {to_target(p) for c in answer.citations for p, _, _, _ in parse_citations(c)}
    sources = {to_target(p) for p in found_paths} | cited
    expected = q["expected_files"]
    total = len(answer.citations) + len(answer.bad_citations)
    if q["type"] == "trap":
        answered = total == 0 and is_refusal(answer.text)
    else:
        answered = bool(answer.citations)
    return {
        "hit_files": sum(f in sources for f in expected) / len(expected) if expected else None,
        "citations_verified": len(answer.citations) / total if total else None,
        "citations": f"{len(answer.citations)}/{total}",
        "answered": answered,
        "latency_s": round(answer.latency_s, 3),
        "tool_calls": len(answer.used_tools),
    }


def _mean(values: list) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def summarize(rows: list[dict]) -> dict:
    def agg(group: list[dict]) -> dict:
        return {"n": len(group),
                "hit_files": _mean([r.get("hit_files") for r in group]),
                "citations_verified": _mean([r.get("citations_verified") for r in group]),
                "answered": _mean([r.get("answered", False) for r in group]),
                "latency_s": _mean([r.get("latency_s") for r in group]),
                "tool_calls": _mean([r.get("tool_calls") for r in group])}
    by_type: dict[str, list[dict]] = {}
    for r in rows:
        by_type.setdefault(r["type"], []).append(r)
    return {"by_type": {t: agg(g) for t, g in by_type.items()}, "overall": agg(rows)}


def run(questions: list[dict], mode: str, fake: bool) -> tuple[list[dict], list[str]]:
    """Прогон агента по вопросам. Возвращает строки результатов и предупреждения об откате на заглушки."""
    from hopiumcode.config import Config
    from hopiumcode.orchestrator import build_default_agent, build_fake_agent, fallback_warnings

    cfg = Config.from_env()
    agent = build_fake_agent(cfg) if fake else build_default_agent(cfg)
    to_target = path_mapper(cfg.repo_root)
    found: set[str] = set()
    inner = agent.retrieve

    def recording_retrieve(question: str, k: int = 8, m: str = "hybrid"):
        hits = inner(question, k, m)
        found.update(h.chunk.path for h in hits)
        return hits

    agent.retrieve = recording_retrieve
    rows = []
    try:
        for q in questions:
            found.clear()
            row = {"id": q["id"], "type": q["type"], "question": q["question"]}
            try:
                ans = agent.answer(q["question"], mode=mode)
            except Exception as ex:  # noqa: BLE001 - ошибка одного вопроса не останавливает прогон
                row.update(error=repr(ex), answered=False)
            else:
                row.update(score(q, ans, found, to_target), answer=ans.text,
                           bad_citations=ans.bad_citations, used_tools=ans.used_tools)
            rows.append(row)
    finally:
        if agent.tools is not None:
            agent.tools.close()
    return rows, fallback_warnings()


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "да" if v else "нет"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def print_report(rows: list[dict], summary: dict, out=None) -> None:
    out = out or sys.stdout
    head = f"{'id':<5} {'тип':<9} {'hit_files':>9} {'ссылки':>7} {'answered':>8} {'сек':>6} {'инстр':>5}"
    print(head, file=out)
    print("-" * len(head), file=out)
    for r in rows:
        if "error" in r:
            print(f"{r['id']:<5} {r['type']:<9} ОШИБКА: {r['error']}", file=out)
            continue
        print(f"{r['id']:<5} {r['type']:<9} {_fmt(r['hit_files']):>9} {r['citations']:>7} "
              f"{_fmt(r['answered']):>8} {_fmt(r['latency_s']):>6} {r['tool_calls']:>5}", file=out)
    head = f"\n{'тип':<9} {'n':>3} {'hit_files':>9} {'cit_ok':>7} {'answered':>8} {'сек':>6} {'инстр':>5}"
    print(head, file=out)
    print("-" * (len(head) - 1), file=out)
    for name, s in [*summary["by_type"].items(), ("ВСЕГО", summary["overall"])]:
        print(f"{name:<9} {s['n']:>3} {_fmt(s['hit_files']):>9} {_fmt(s['citations_verified']):>7} "
              f"{_fmt(s['answered']):>8} {_fmt(s['latency_s']):>6} {_fmt(s['tool_calls']):>5}", file=out)


def _target_commit() -> str | None:
    res = subprocess.run(["git", "-C", str(TARGET_ROOT), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=False)
    return res.stdout.strip() or None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="run_eval", description="Оценка агента по eval/questions.yaml")
    p.add_argument("--mode", default="hybrid", choices=MODES)
    p.add_argument("--fake", action="store_true", help="заглушки вместо Ollama, ретривера и MCP")
    p.add_argument("--only", default="", help="id вопросов через запятую, например q01,q05")
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = p.parse_args(argv)

    questions = load_questions()
    if args.only:
        wanted = [x.strip() for x in args.only.split(",") if x.strip()]
        unknown = set(wanted) - {q["id"] for q in questions}
        if unknown:
            print(f"Нет вопросов с id: {', '.join(sorted(unknown))}", file=sys.stderr)
            return 1
        questions = [q for q in questions if q["id"] in wanted]

    started = datetime.now()
    rows, warnings = run(questions, args.mode, args.fake)
    summary = summarize(rows)
    result = {"started": started.isoformat(timespec="seconds"), "mode": args.mode, "fake": args.fake,
              "target_commit": _target_commit(), "fallbacks": warnings, "summary": summary, "questions": rows}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{started:%Y%m%d-%H%M%S}_{args.mode}{'_fake' if args.fake else ''}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Режим: {args.mode}{' (fake)' if args.fake else ''}, вопросов: {len(rows)}")
    print_report(rows, summary)
    if args.fake:
        print("\n--fake: модель ScriptedChat, ретривер stub_retrieve, инструменты StubTools (без Ollama);"
              " метрики проверяют цепочку, а не качество ответов.")
    if warnings:
        print("\nЗаглушки вместо модулей участников:")
        for w in warnings:
            print(f"  {w}")
    print(f"\nРезультат: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
