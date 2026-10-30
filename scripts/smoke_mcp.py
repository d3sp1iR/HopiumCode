"""Ручная проверка связки MCP-сервер + McpClient (+ выбор инструментов моделью).

Запуск из корня проекта, с активированным .venv:

    export REPO_ROOT="$PWD/data/repo/flask/src/flask"
    python scripts/smoke_mcp.py                       # сервер и клиент, без модели
    python scripts/smoke_mcp.py --ollama qwen3:8b     # + какой инструмент выберет модель
"""

import argparse
import re
import sys
import time

from hopiumcode.mcp_client import McpClient, McpClientError

GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"


def check(title: str, ok: bool, detail: str) -> bool:
    mark = f"{GREEN}OK  {RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"[{mark}] {title}")
    print("       " + detail.replace("\n", "\n       ")[:600])
    return ok


def run_basic(c: McpClient, file: str) -> bool:
    tools = c.list_tools_for_ollama()
    names = [t["function"]["name"] for t in tools]
    results = [check("5 инструментов", len(names) == 5, ", ".join(names))]

    out = c.call_tool("read_file", {"path": file, "end_line": 5})
    results.append(check("read_file читает", out.startswith(" 1|"), out))

    out = c.call_tool("read_file", {"path": file, "start_line": "3", "end_line": "4"})
    results.append(check("read_file: строка '3' → число 3", out.startswith(" 3|"), out))

    out = c.call_tool("read_file", {"path": "../../../../etc/passwd"})
    results.append(check("read_file: выход из репозитория запрещён",
                         out.startswith("Ошибка") and "вне репозитория" in out, out))

    out = c.call_tool("read_file", {"path": "/etc/passwd"})
    results.append(check("read_file: абсолютный путь запрещён", "вне репозитория" in out, out))

    out = c.call_tool("read_file", {"path": "нет_такого.py"})
    results.append(check("read_file: несуществующий файл", "не найден" in out, out))

    out = c.call_tool("git_log", {"path": file, "n": 3})
    row = re.compile(r"^[0-9a-f]{7} \| .+ \| \d{4}-\d\d-\d\dT.+ \| .+$")
    results.append(check("git_log", all(row.match(r) for r in out.splitlines()) and out != "",
                         out))

    out = c.call_tool("git_blame", {"path": file, "line": 1})
    results.append(check("git_blame", out.startswith(f"{file}:1 | ") and "\n 1|" in out, out))

    out = c.call_tool("git_blame", {"path": file, "line": 999999})
    results.append(check("git_blame: строка вне диапазона", "вне диапазона" in out, out))

    out = c.call_tool("search_code", {"query": "session cookie"})
    print(f"[инфо] search_code: {out[:200]}")  # пока заглушка — это нормально
    return all(results)


# ТЗ, раздел 10: на 10 вопросах модель должна выбрать нужный инструмент не менее 7 раз
QUESTIONS = [
    ("Покажи первые 20 строк файла {file}", "read_file"),
    ("Процитируй строки 30–50 из файла {file}", "read_file"),
    ("В каком месте проекта подписываются cookie сессии?", "search_code"),
    ("Где обрабатывается ошибка 404?", "search_code"),
    ("Кто вызывает функцию open_session?", "find_callers"),
    ("Что сломается, если поменять метод make_response?", "find_callers"),
    ("Какие коммиты меняли файл {file}?", "git_log"),
    ("Кто и когда в последний раз менял файл {file}?", "git_log"),
    ("Кто и когда последним менял строку 1 в файле {file}?", "git_blame"),
    ("В каком коммите появилась строка 10 файла {file}?", "git_blame"),
]
PASS_THRESHOLD = 7


def run_ollama(c: McpClient, model: str, file: str) -> bool:
    try:
        import ollama
    except ImportError:
        print("пакет ollama не установлен: pip install ollama")
        return False
    tools = c.list_tools_for_ollama()
    system = ("Ты помощник по коду репозитория Flask. Отвечай, только опираясь на "
              "результаты инструментов, и сначала вызови подходящий инструмент.")
    good = 0
    for template, expected in QUESTIONS:
        question = template.format(file=file)
        t0 = time.monotonic()
        resp = ollama.chat(model=model, tools=tools, messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ])
        calls = resp.message.tool_calls or []
        chosen = calls[0].function.name if calls else "(без инструмента)"
        args = dict(calls[0].function.arguments) if calls else {}
        ok = chosen == expected
        good += ok
        detail = f"выбрал {chosen}({args}), ждали {expected}, {time.monotonic() - t0:.1f} с"
        if calls:
            detail += "\nрезультат: " + c.call_tool(chosen, args)[:200]
        check(question, ok, detail)
    verdict = "критерий ТЗ выполнен" if good >= PASS_THRESHOLD else "НИЖЕ критерия ТЗ (7/10)"
    print(f"\nВыбор инструмента: {good}/{len(QUESTIONS)} — {verdict}")
    return good >= PASS_THRESHOLD


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="app.py", help="файл внутри REPO_ROOT для проверок")
    ap.add_argument("--ollama", metavar="MODEL", help="проверить выбор инструментов моделью")
    a = ap.parse_args()
    try:
        with McpClient() as c:
            print(f"сервер: {c.server_info}\n")
            ok = run_basic(c, a.file)
            if a.ollama:
                print()
                ok = run_ollama(c, a.ollama, a.file) and ok
    except McpClientError as exc:
        print(f"{RED}Сервер не запустился:{RESET} {exc}")
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
