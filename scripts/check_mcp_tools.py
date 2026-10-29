"""Проверка выбора MCP-инструментов моделью. Владелец: участник 4.

python scripts/check_mcp_tools.py [--model qwen3:4b] [--mode agent|bare] [--questions eval/questions.yaml]
                                  [--ids q01,q05,...]

Модель получает системный промпт агента и реальный список инструментов с MCP-сервера (McpClient), а в
сообщении пользователя: --mode agent — первый запрос оркестратора (ENRICHED_TEMPLATE с пустым контекстом и
подсказкой по типу вопроса, как при ещё не готовом ретривере); --mode bare — только сам вопрос. Печатается первый вызванный инструмент, ожидался ли он, сработал ли вызов
с аргументами модели и время ответа. Ожидание: semantic/overview — search_code или read_file,
graph — find_callers, history — git_log или git_blame. Нужны Ollama и клон Flask (make fetch).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

EXPECTED = {"semantic": {"search_code", "read_file"}, "overview": {"search_code", "read_file"},
            "graph": {"find_callers"}, "history": {"git_log", "git_blame"}}
DEFAULT_IDS = "q01,q02,q03,q05,q20,q22,q08,q09,q10,q11"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3:4b")
    p.add_argument("--questions", type=Path, default=ROOT / "eval" / "questions.yaml")
    p.add_argument("--ids", default=DEFAULT_IDS)
    p.add_argument("--mode", choices=["agent", "bare"], default="agent")
    args = p.parse_args()

    import ollama

    from eval.run_eval import load_questions
    from hopiumcode.config import Config
    from hopiumcode.mcp_client import McpClient
    from hopiumcode.orchestrator import _HINTS, OllamaChat, classify
    from hopiumcode.prompts import ENRICHED_TEMPLATE, SYSTEM_PROMPT

    questions = {q["id"]: q for q in load_questions(args.questions)}
    ids = [i.strip() for i in args.ids.split(",") if i.strip()]
    missing = [i for i in ids if i not in questions]
    if missing:
        print(f"Нет вопросов {', '.join(missing)} в {args.questions}, пропускаю", file=sys.stderr)
    cfg = Config.from_env()
    chat = OllamaChat(args.model, cfg.ollama_url)  # temperature 0
    system = SYSTEM_PROMPT.format(repo_name="Flask")
    ok_count = total = 0
    with McpClient() as client:
        tools = client.list_tools_for_ollama()
        print(f"Модель {args.model}, режим {args.mode}, инструментов: {len(tools)}\n")
        print("| id | тип | ожидалось | вызвано | верно | вызов сработал | сек |")
        print("| --- | --- | --- | --- | --- | --- | --- |")
        for qid in [i for i in ids if i in questions]:
            q = questions[qid]
            t = time.perf_counter()
            user = q["question"]
            if args.mode == "agent":  # как Agent.answer при пустой выдаче ретривера
                user = ENRICHED_TEMPLATE.format(question=q["question"], context_blocks="(ничего не найдено)",
                                                graph_facts="(нет)")
                hint = _HINTS[classify(q["question"])]
                user += f"\n\n{hint}" if hint else ""
            try:
                reply = chat.chat([{"role": "system", "content": system}, {"role": "user", "content": user}], tools)
            except (ConnectionError, ollama.ResponseError) as ex:
                print(f"Ollama недоступна или нет модели {args.model}: {ex}", file=sys.stderr)
                return 2
            dt = time.perf_counter() - t
            call = reply.tool_calls[0] if reply.tool_calls else None
            expected = EXPECTED.get(q["type"], set())
            good = bool(call and call.name in expected)
            if call:
                out = client.call_tool(call.name, call.args)
                worked = "нет: " + out.splitlines()[0][:60] if out.startswith("Ошибка") else "да"
                called = f"{call.name}({', '.join(f'{k}={v!r}' for k, v in call.args.items())})"
            else:
                worked, called = "—", "без инструмента"
            total += 1
            ok_count += good
            print(f"| {qid} | {q['type']} | {' / '.join(sorted(expected))} | `{called}` | "
                  f"{'да' if good else '**нет**'} | {worked} | {dt:.1f} |")
    print(f"\nВерный инструмент: {ok_count} из {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
