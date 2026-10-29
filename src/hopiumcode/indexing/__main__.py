"""python -m hopiumcode.indexing build [--full] — сборка индекса. Владелец: участник 2."""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m hopiumcode.indexing")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="построить или обновить индекс (векторы, BM25, граф)")
    b.add_argument("--full", action="store_true", help="пересобрать всё, не глядя на хэши файлов")
    b.add_argument("--no-graph", action="store_true", help="не строить граф")
    args = p.parse_args(argv)

    from ..config import Config
    from .build import build_all
    from .embed import EmbeddingError

    cfg = Config.from_env()
    root = cfg.repo_root
    if not root.is_dir() or not any(root.rglob("*.py")):
        print(f"Нет исходников для индексации в {root}. Сначала выполните: make fetch", file=sys.stderr)
        return 1
    print(f"Индексация {root} -> {cfg.index_dir} (эмбеддинги: {cfg.embed_model})")
    try:
        stats = build_all(config=cfg, full=args.full, graph=not args.no_graph)
    except EmbeddingError as ex:
        print(f"Индекс не достроен: {ex}\nУже обработанные файлы сохранены, повторный запуск продолжит.",
              file=sys.stderr)
        return 2
    print(stats.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
