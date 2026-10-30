"""Консоль. Владелец: участник 1.

python -m hopiumcode.cli ask "вопрос" [--mode hybrid] [--fake] [-v]
python -m hopiumcode.cli index

Модули участников, которые ещё бросают NotImplementedError, заменяются заглушками из stubs.py
с предупреждением в stderr (см. orchestrator.warn_fallback).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _has_sources(root: Path) -> bool:
    return root.is_dir() and any(root.rglob("*.py"))


def index(root: Path) -> int:
    if not _has_sources(root):
        print(f"Нет исходников для индексации в {root}. Сначала выполните: make fetch", file=sys.stderr)
        return 1
    from .indexing.build import build_all
    from .orchestrator import warn_fallback
    try:
        build_all()
    except ConnectionError as ex:
        print(f"Ollama недоступна: {ex}\nЗапустите `ollama serve` и скачайте модель эмбеддингов "
              f"(`ollama pull nomic-embed-text`).", file=sys.stderr)
        return 2
    except NotImplementedError as ex:
        from .stubs import stub_build_all
        warn_fallback("indexing.build.build_all", ex, "stubs.stub_build_all")
        stub_build_all(root)
    return 0


def ask(question: str, mode: str, fake: bool) -> int:
    from .config import Config
    from .orchestrator import build_default_agent, build_fake_agent
    cfg = Config.from_env()
    agent = build_fake_agent(cfg) if fake else build_default_agent(cfg)
    try:
        res = agent.answer(question, mode=mode)
    except ConnectionError as ex:
        print(f"Ollama недоступна ({cfg.ollama_url}): {ex}\nЗапустите `ollama serve` или используйте --fake.",
              file=sys.stderr)
        return 2
    except Exception as ex:
        from .indexing.vectorstore import IndexNotBuilt
        if isinstance(ex, (IndexNotBuilt, FileNotFoundError)):
            print(f"{ex}\nСначала выполните: make index", file=sys.stderr)
            return 3
        if type(ex).__module__.split(".")[0] != "ollama":  # например, модель не скачана
            raise
        print(f"Ollama: {ex}\nСкачайте модель (`ollama pull {cfg.llm_model}`) или задайте CG_LLM_MODEL.",
              file=sys.stderr)
        return 2
    finally:
        if agent.tools is not None:
            agent.tools.close()
    print(res.text)
    print(f"\n[шагов модели: {res.steps}; инструменты: {', '.join(res.used_tools) or 'нет'}; {res.latency_s:.1f} c]")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hopiumcode")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask", help="задать вопрос агенту")
    a.add_argument("question")
    a.add_argument("--mode", default="hybrid", choices=["vector", "vector+bm25", "hybrid"])
    a.add_argument("--fake", action="store_true", help="заглушки вместо Ollama, ретривера и MCP")
    a.add_argument("-v", "--verbose", action="store_true", help="журнал шагов в stderr")
    sub.add_parser("index", help="проиндексировать репозиторий (векторы и граф)")
    args = p.parse_args(argv)

    if args.cmd == "ask":
        if args.verbose:
            logging.basicConfig(level=logging.INFO, format="%(message)s")
        return ask(args.question, args.mode, args.fake)
    from .config import Config
    return index(Config.from_env().repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
