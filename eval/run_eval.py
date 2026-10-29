"""Сравнение режимов retrieve: vector | vector+bm25 | hybrid. Владелец: участник 5.

python -m eval.run_eval [--mode hybrid|vector|vector+bm25|all] [--fake] [--only q01,q05] [--limit N]
                        [--judge] [--out-dir eval/results]

Метрики на вопрос:
  recall@5, recall@8  доля expected_files среди файлов первых k хитов retrieve (пути вида src/flask/...)
  hit_files           доля expected_files среди найденных фрагментов и подтверждённых ссылок ответа
  citations_verified  подтверждено ссылок / всего ссылок в ответе (None, если ссылок нет)
  answered            trap: агент сказал, что не нашёл, и не дал ни одной ссылки;
                      остальные типы: есть хотя бы одна подтверждённая ссылка
  latency_s, steps, tool_calls
  judge               (--judge) оценка LLM-судьи 0/1/2 по эталону, промпт П5.3

Модель и судья вызываются с температурой 0 (OllamaChat). Режим, который модули участников ещё
не реализуют (NotImplementedError), помечается «недоступно (заглушка)». Результат:
eval/results/<дата-время>.json и .md.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = Path(__file__).with_name("questions.yaml")
RESULTS_DIR = Path(__file__).with_name("results")
TARGET_ROOT = ROOT / "data" / "repo" / "flask"  # от него считаются пути в questions.yaml
MODES = ("vector", "vector+bm25", "hybrid")
RECALL_KS = (5, 8)
UNAVAILABLE = "недоступно (заглушка)"
JUDGE_PROMPT = (
    "Ты проверяешь ответ помощника по коду. Оцени: 2 — по существу совпадает с эталоном и не содержит "
    "выдуманных фактов; 1 — частично верен или неполон; 0 — неверен или содержит выдуманные имена и "
    'ссылки. Верни JSON {"score": 0|1|2, "reason": "одно предложение"}. Опирайся только на эталон.'
)
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


def recall_at_k(expected: list[str], ranked_paths: list[str], k: int) -> float | None:
    """Доля expected среди уникальных файлов первых k хитов. None, если эталонных файлов нет (trap)."""
    if not expected:
        return None
    top = set(ranked_paths[:k])
    return sum(f in top for f in expected) / len(expected)


def score(q: dict, answer, ranked_paths: Iterable[str], to_target: Callable[[str], str]) -> dict:
    """Метрики одного ответа (Answer из models.py) по эталону вопроса.

    ranked_paths: пути хитов retrieve в порядке выдачи (относительно Config.repo_root).
    """
    from hopiumcode.orchestrator import parse_citations

    ranked = [to_target(p) for p in ranked_paths]
    cited = {to_target(p) for c in answer.citations for p, _, _, _ in parse_citations(c)}
    sources = set(ranked) | cited
    expected = q["expected_files"]
    total = len(answer.citations) + len(answer.bad_citations)
    if q["type"] == "trap":
        answered = total == 0 and is_refusal(answer.text)
    else:
        answered = bool(answer.citations)
    return {
        **{f"recall@{k}": recall_at_k(expected, ranked, k) for k in RECALL_KS},
        "hit_files": sum(f in sources for f in expected) / len(expected) if expected else None,
        "citations_verified": len(answer.citations) / total if total else None,
        "citations": f"{len(answer.citations)}/{total}",
        "answered": answered,
        "latency_s": round(answer.latency_s, 3),
        "steps": answer.steps,
        "tool_calls": len(answer.used_tools),
        "found_files": list(dict.fromkeys(ranked)),
    }


# --------------------------------------------------------------------------- судья

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_judge(text: str) -> dict:
    """Разбор ответа судьи: {"score": 0|1|2 или None, "reason": str, "error": str (если не разобрался)}."""
    m = _JSON_OBJ_RE.search(text or "")
    if not m:
        return {"score": None, "reason": None, "error": "в ответе судьи нет JSON"}
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as ex:
        return {"score": None, "reason": None, "error": f"JSON не разобран: {ex}"}
    sc = data.get("score") if isinstance(data, dict) else None
    if isinstance(sc, str) and sc.strip().isdigit():
        sc = int(sc.strip())
    if isinstance(sc, bool) or sc not in (0, 1, 2):
        return {"score": None, "reason": None, "error": f"недопустимый score: {sc!r}"}
    return {"score": sc, "reason": str(data.get("reason", ""))}


def judge(q: dict, answer_text: str, chat: Callable[[list[dict]], str]) -> dict:
    """Оценка ответа LLM-судьёй по эталону. Ошибки модели и разбора не роняют прогон."""
    user = (f"Вопрос: {q['question']}\n\nЭталон: {q['reference_answer']}\n\n"
            f"Ответ помощника:\n{answer_text}")
    try:
        text = chat([{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": user}])
    except Exception as ex:  # noqa: BLE001 - судья вспомогательный
        return {"score": None, "reason": None, "error": f"судья недоступен: {ex!r}"}
    return parse_judge(text)


def _make_judge_chat(cfg, fake: bool) -> Callable[[list[dict]], str]:
    if fake:
        def fake_chat(messages: list[dict]) -> str:
            raise RuntimeError("--fake: судья без Ollama не вызывается")  # score None, а не выдуманная оценка
        return fake_chat
    from hopiumcode.orchestrator import OllamaChat
    model = OllamaChat(cfg.llm_model, cfg.ollama_url)  # temperature 0
    return lambda messages: model.chat(messages, None).content


# --------------------------------------------------------------------------- прогон

def _mean(values: list) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


_METRICS = ("recall@5", "recall@8", "hit_files", "citations_verified", "answered", "latency_s", "steps",
            "tool_calls", "judge")


def summarize(rows: list[dict]) -> dict:
    """Средние по типам вопросов и в целом. Строки недоступного режима в средние не входят."""
    def agg(group: list[dict]) -> dict:
        done = [r for r in group if not r.get("unavailable")]
        res = {"n": len(group)}
        for m in _METRICS:
            default = False if m == "answered" else None
            res[m] = _mean([r.get(m, default) for r in done]) if done else None
        return res
    by_type: dict[str, list[dict]] = {}
    for r in rows:
        by_type.setdefault(r["type"], []).append(r)
    return {"by_type": {t: agg(g) for t, g in by_type.items()}, "overall": agg(rows)}


def retriever_status(mode: str, fake: bool) -> str:
    """Статус настоящего retrieval.hybrid.retrieve для режима: "есть", UNAVAILABLE или "ошибка: ..."."""
    from hopiumcode.retrieval.hybrid import retrieve
    try:
        retrieve("проверка режима", 1, mode)
    except NotImplementedError:
        return UNAVAILABLE
    except Exception as ex:  # noqa: BLE001 - например, нет индекса или Ollama в --fake
        return f"ошибка: {ex!r}"[:120]
    return "есть"


def run(questions: list[dict], mode: str, fake: bool, agent=None, judge_chat=None) -> tuple[list[dict], list[str]]:
    """Прогон агента по вопросам в одном режиме. Возвращает строки и предупреждения об откате на заглушки."""
    from hopiumcode.config import Config
    from hopiumcode.orchestrator import build_default_agent, build_fake_agent, fallback_warnings

    cfg = Config.from_env()
    own = agent is None
    if own:
        agent = build_fake_agent(cfg) if fake else build_default_agent(cfg)
    to_target = path_mapper(cfg.repo_root)
    ranked: list[str] = []
    inner = agent.retrieve

    def recording_retrieve(question: str, k: int = 8, m: str = "hybrid"):
        hits = inner(question, k, m)
        ranked.extend(h.chunk.path for h in hits)
        return hits

    agent.retrieve = recording_retrieve
    rows = []
    try:
        for q in questions:
            ranked.clear()
            row = {"id": q["id"], "type": q["type"], "mode": mode, "question": q["question"]}
            try:
                ans = agent.answer(q["question"], mode=mode)
            except Exception as ex:  # noqa: BLE001 - ошибка одного вопроса не останавливает прогон
                row.update(error=repr(ex), answered=False)
            else:
                row.update(score(q, ans, ranked, to_target), answer=ans.text,
                           bad_citations=ans.bad_citations, used_tools=ans.used_tools)
                if judge_chat is not None:
                    verdict = judge(q, ans.text, judge_chat)
                    row.update(judge=verdict["score"], judge_reason=verdict.get("reason"),
                               **({"judge_error": verdict["error"]} if verdict.get("error") else {}))
            rows.append(row)
    finally:
        agent.retrieve = inner
        if own and agent.tools is not None:
            agent.tools.close()
    return rows, fallback_warnings()


def unavailable_rows(questions: list[dict], mode: str) -> list[dict]:
    return [{"id": q["id"], "type": q["type"], "mode": mode, "question": q["question"], "unavailable": True}
            for q in questions]


# --------------------------------------------------------------------------- отчёт

def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "да" if v else "нет"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


_COLS = [("recall@5", "r@5"), ("recall@8", "r@8"), ("citations_verified", "cit_ok"), ("answered", "answered"),
         ("latency_s", "сек"), ("steps", "шагов")]


def _cols(with_judge: bool) -> list[tuple[str, str]]:
    return _COLS + ([("judge", "судья")] if with_judge else [])


def mode_type_table(summaries: dict[str, dict], statuses: dict[str, str], with_judge: bool = False) -> list[list[str]]:
    """Строки таблицы «режим x тип»: [режим, тип, n, метрики...]. Недоступный режим помечается."""
    rows = []
    for mode, summ in summaries.items():
        for typ, s in summ["by_type"].items():
            if statuses.get(mode) == UNAVAILABLE and s["recall@5"] is None and s["answered"] is None:
                rows.append([mode, typ, str(s["n"])] + [UNAVAILABLE] + [""] * (len(_cols(with_judge)) - 1))
            else:
                rows.append([mode, typ, str(s["n"])] + [_fmt(s[m]) for m, _ in _cols(with_judge)])
    return rows


def mode_total_table(summaries: dict[str, dict], statuses: dict[str, str], with_judge: bool = False) -> list[list[str]]:
    rows = []
    for mode, summ in summaries.items():
        s = summ["overall"]
        rows.append([mode, statuses.get(mode, "-"), str(s["n"])] + [_fmt(s[m]) for m, _ in _cols(with_judge)])
    return rows


def format_table(header: list[str], rows: list[list[str]]) -> str:
    """Таблица с выравниванием по ширине колонок (первые две влево, остальные вправо)."""
    widths = [max(len(str(c)) for c in col) for col in zip(header, *rows)]

    def line(cells):
        return "  ".join(str(c).ljust(w) if i < 2 else str(c).rjust(w) for i, (c, w) in enumerate(zip(cells, widths)))
    return "\n".join([line(header), "  ".join("-" * w for w in widths), *map(line, rows)])


def markdown_table(header: list[str], rows: list[list[str]]) -> str:
    def esc(c) -> str:
        return str(c).replace("|", "\\|").replace("\n", " ")
    out = ["| " + " | ".join(map(esc, header)) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(map(esc, r)) + " |" for r in rows]
    return "\n".join(out)


def weakest(rows: list[dict], n: int = 5) -> list[dict]:
    """Самые слабые вопросы режима: сначала без ответа, затем по recall@8, hit_files и доле ссылок."""
    done = [r for r in rows if not r.get("unavailable")]

    def key(r):
        def num(m):
            v = r.get(m)
            return -1.0 if v is None else float(v)
        return (bool(r.get("answered")), num("recall@8"), num("hit_files"), num("citations_verified"))
    return sorted(done, key=key)[:n]


def _expected_text(q: dict) -> str:
    if q["type"] == "trap":
        return "честное «не нашёл», без ссылок"
    return ", ".join(q["expected_files"])


def _got_text(r: dict) -> str:
    if "error" in r:
        return f"ошибка: {r['error']}"
    files = ", ".join(r.get("found_files", [])[:5]) or "ничего"
    answer = " ".join((r.get("answer") or "").split())[:160]
    return f"найдено: {files}; ответ: {answer}"


def render_markdown(result: dict, questions: list[dict]) -> str:
    by_id = {q["id"]: q for q in questions}
    wj = result["judge"]
    cols = [h for _, h in _cols(wj)]
    md = [f"# Оценка агента: {result['started']}", "",
          f"- Режимы: {', '.join(result['modes'])}{' (--fake: заглушки вместо модели, ретривера и MCP)' if result['fake'] else ''}",
          f"- Вопросов: {len(questions)}; коммит Flask: {result['target_commit'] or 'неизвестен'}",
          f"- Судья: {'включён' if wj else 'выключен'}", "",
          "## Режим x тип вопроса", "",
          markdown_table(["режим", "тип", "n", *cols], mode_type_table(result["summary"], result["retriever_status"], wj)),
          "", "## Итог по режимам", "",
          markdown_table(["режим", "ретривер", "n", *cols], mode_total_table(result["summary"], result["retriever_status"], wj)),
          ""]
    for mode in result["modes"]:
        md += [f"## 5 самых слабых вопросов: {mode}", ""]
        weak = weakest(result["questions"][mode])
        if not weak:
            md += [f"Режим {UNAVAILABLE}: вопросы не прогонялись." if result["retriever_status"].get(mode) == UNAVAILABLE
                   else "Нет данных.", ""]
            continue
        md.append(markdown_table(["id", "вопрос", "ожидалось", "получено"],
                                 [[r["id"], r["question"], _expected_text(by_id[r["id"]]), _got_text(r)] for r in weak]))
        md.append("")
    if result["fallbacks"]:
        md += ["## Заглушки вместо модулей участников", "", *[f"- {w}" for w in result["fallbacks"]], ""]
    return "\n".join(md)


def print_question_table(rows: list[dict], out=None) -> None:
    out = out or sys.stdout
    header = ["id", "тип", "r@5", "r@8", "ссылки", "answered", "сек", "шагов", "инстр"]
    body = []
    for r in rows:
        if r.get("unavailable"):
            body.append([r["id"], r["type"], UNAVAILABLE, "", "", "", "", "", ""])
        elif "error" in r:
            body.append([r["id"], r["type"], "ОШИБКА", r["error"][:40], "", "", "", "", ""])
        else:
            body.append([r["id"], r["type"], _fmt(r["recall@5"]), _fmt(r["recall@8"]), r["citations"],
                         _fmt(r["answered"]), _fmt(r["latency_s"]), str(r["steps"]), str(r["tool_calls"])])
    print(format_table(header, body), file=out)


def print_report(result: dict, out=None) -> None:
    out = out or sys.stdout
    wj = result["judge"]
    cols = [h for _, h in _cols(wj)]
    if len(result["modes"]) == 1:
        print_question_table(result["questions"][result["modes"][0]], out)
        print(file=out)
    print("Режим x тип вопроса", file=out)
    print(format_table(["режим", "тип", "n", *cols], mode_type_table(result["summary"], result["retriever_status"], wj)),
          file=out)
    print("\nИтог по режимам", file=out)
    print(format_table(["режим", "ретривер", "n", *cols],
                       mode_total_table(result["summary"], result["retriever_status"], wj)), file=out)


def _target_commit() -> str | None:
    res = subprocess.run(["git", "-C", str(TARGET_ROOT), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=False)
    return res.stdout.strip() or None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="run_eval", description="Оценка агента по eval/questions.yaml")
    p.add_argument("--mode", default="hybrid", choices=[*MODES, "all"])
    p.add_argument("--fake", action="store_true", help="заглушки вместо Ollama, ретривера и MCP")
    p.add_argument("--only", default="", help="id вопросов через запятую, например q01,q05")
    p.add_argument("--limit", type=int, default=0, help="взять только первые N вопросов")
    p.add_argument("--judge", action="store_true", help="LLM-судья 0/1/2 по эталону (та же модель Ollama)")
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
    if args.limit > 0:
        questions = questions[:args.limit]

    from hopiumcode.config import Config
    from hopiumcode.orchestrator import build_default_agent, build_fake_agent

    cfg = Config.from_env()
    modes = list(MODES) if args.mode == "all" else [args.mode]
    statuses = {m: retriever_status(m, args.fake) for m in modes}
    agent = build_fake_agent(cfg) if args.fake else build_default_agent(cfg)
    judge_chat = _make_judge_chat(cfg, args.fake) if args.judge else None
    started = datetime.now()
    per_mode: dict[str, list[dict]] = {}
    warnings: list[str] = []
    try:
        for m in modes:
            if statuses[m] == UNAVAILABLE and not args.fake:
                per_mode[m] = unavailable_rows(questions, m)  # прогон на заглушке ничего не скажет о режиме
                continue
            per_mode[m], warnings = run(questions, m, args.fake, agent=agent, judge_chat=judge_chat)
    finally:
        if agent.tools is not None:
            agent.tools.close()

    result = {"started": started.isoformat(timespec="seconds"), "modes": modes, "fake": args.fake,
              "judge": args.judge, "target_commit": _target_commit(), "retriever_status": statuses,
              "fallbacks": warnings, "summary": {m: summarize(r) for m, r in per_mode.items()},
              "questions": per_mode}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"{started:%Y%m%d-%H%M%S}"
    n = 1
    while stem.with_suffix(".json").exists():  # два прогона в одну секунду не затирают друг друга
        stem = args.out_dir / f"{started:%Y%m%d-%H%M%S}-{n}"
        n += 1
    json_path, md_path = stem.with_suffix(".json"), stem.with_suffix(".md")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(result, questions), encoding="utf-8")

    print(f"Режимы: {', '.join(modes)}{' (fake)' if args.fake else ''}, вопросов: {len(questions)}\n")
    print_report(result)
    if args.fake:
        print("\n--fake: модель ScriptedChat, ретривер stub_retrieve, инструменты StubTools (без Ollama);"
              " метрики проверяют цепочку, а не качество ответов. Колонка «ретривер» — статус настоящего модуля.")
    if warnings:
        print("\nЗаглушки вместо модулей участников:")
        for w in warnings:
            print(f"  {w}")
    print(f"\nРезультат: {json_path}\nОтчёт: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
