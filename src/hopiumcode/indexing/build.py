"""Полный процесс индексации репозитория. Владелец: участник 2 (граф: участник 3)."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from ..config import Config
from ..graph.build import build_graph
from .chunker import chunk_file
from .loader import iter_files

log = logging.getLogger(__name__)

MANIFEST = "manifest.json"
CHUNKER_VERSION = 2  # увеличить при изменении правил разбиения: индекс пересоберётся целиком


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_manifest(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def build_all(config: Config | None = None, index: Any | None = None, graph: bool = True) -> dict[str, Any]:
    """Индексирует репозиторий: векторы (инкрементально по sha256 файлов) и граф кода.

    Возвращает статистику: файлов всего/изменено/удалено, фрагментов добавлено, секунд.
    index и graph нужны тестам: можно подставить индекс-заглушку и не строить граф.
    """
    cfg = config or Config.from_env()
    started = time.monotonic()
    root = Path(cfg.repo_root)
    if not root.is_dir():
        raise FileNotFoundError(f"нет каталога репозитория: {root}. Сначала выполните `make fetch`")
    cfg.index_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(cfg.index_dir) / MANIFEST

    if index is None:
        from .vectorstore import VectorIndex
        index = VectorIndex(cfg)

    old = _load_manifest(manifest_path)
    meta = {"embed_model": cfg.embed_model, "root": str(root.resolve()), "chunker_version": CHUNKER_VERSION}
    full = any(old.get(k) != v for k, v in meta.items())
    old_files: dict[str, str] = {} if full else old.get("files", {})

    current: dict[str, str] = {}
    files: dict[str, Path] = {}
    for path in iter_files(root):
        rel = path.relative_to(root).as_posix()
        current[rel] = _sha256(path)
        files[rel] = path

    removed = [p for p in old_files if p not in current]
    changed = [p for p, h in current.items() if old_files.get(p) != h]
    if full and old:
        log.info("модель эмбеддингов, каталог или правила разбиения изменились: полная переиндексация")

    if full:
        index.reset()                          # другая модель или каталог: старые векторы не годятся
    else:
        for rel in removed + [p for p in changed if p in old_files]:
            index.delete_by_path(rel)

    for rel in removed:
        old_files.pop(rel, None)
    _save_manifest(manifest_path, {**meta, "files": old_files})
    added = 0
    for rel in changed:
        chunks = chunk_file(files[rel], root)
        if chunks:
            index.add(chunks)
        added += len(chunks)
        old_files[rel] = current[rel]
        _save_manifest(manifest_path, {**meta, "files": old_files})  # по файлам: сбой не теряет прогресс

    stats: dict[str, Any] = {"files": len(current), "changed": len(changed), "removed": len(removed),
                             "chunks_added": added}
    if graph:
        stats["graph"] = build_graph(cfg)
    stats["seconds"] = round(time.monotonic() - started, 2)
    print(f"Индекс: файлов {stats['files']}, изменено {stats['changed']}, удалено {stats['removed']}, "
          f"фрагментов добавлено {stats['chunks_added']}, {stats['seconds']} с")
    if graph:
        g = stats["graph"]
        print(f"Граф: узлы {g['nodes']}, рёбра {g['edges']}")
    return stats
