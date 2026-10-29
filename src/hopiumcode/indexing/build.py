"""Команда index: векторы (участник 2) и граф (участник 3). Владелец: участник 2.

python -m hopiumcode.indexing build [--full]   (make index)

Инкрементально: sha256 каждого файла хранится в Config.index_dir / "file_hashes.json"; неизменённые файлы
пропускаются, удалённые убираются из индекса. Если сменилась модель эмбеддингов, корень репозитория или
версия разбиения (CHUNKER_VERSION), индекс пересобирается целиком. BM25 строится по фрагментам из векторного
индекса, поэтому оба индекса согласованы. Граф строится, если graph.extract реализован (участник 3).
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Config
from .chunker import chunk_file
from .loader import iter_files

HASHES_FILE = "file_hashes.json"
CHUNKER_VERSION = 1  # увеличить при изменении логики chunker: индекс пересоберётся


@dataclass
class BuildStats:
    files: int = 0
    indexed_files: int = 0        # переиндексированы (новые или изменённые)
    unchanged_files: int = 0      # пропущены как неизменённые
    removed_files: int = 0
    chunks: int = 0               # всего фрагментов в индексе
    new_chunks: int = 0           # отправлено на эмбеддинг в этом запуске
    seconds: float = 0.0
    full_rebuild: bool = False
    graph: str = "не строился"
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (f"Файлов: {self.files} (переиндексировано {self.indexed_files}, без изменений "
                f"{self.unchanged_files}, удалено {self.removed_files}); фрагментов в индексе: {self.chunks} "
                f"(новых {self.new_chunks}); {self.seconds:.1f} c"
                f"{'; полная пересборка' if self.full_rebuild else ''}. Граф: {self.graph}.")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_state(file: Path) -> dict:
    if file.exists():
        try:
            return json.loads(file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def _save_state(file: Path, state: dict) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    tmp = file.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(file)


def _build_graph(root: Path, cfg: Config) -> str:
    """Граф кода (участник 3); пока graph.extract не реализован, просто сообщаем об этом."""
    from ..graph.extract import extract
    from ..graph.store import KuzuStore, NetworkXStore
    try:
        nodes, edges = extract(root)
        store = KuzuStore() if cfg.graph_backend == "kuzu" else NetworkXStore()
        store.add_nodes(nodes)
        store.add_edges(edges)
        store.save()
    except NotImplementedError as ex:
        return f"не реализован ({ex})"
    except Exception as ex:  # noqa: BLE001 - ошибка графа не должна ломать векторный индекс
        return f"ошибка: {ex!r}"
    return f"узлов {len(nodes)}, рёбер {len(edges)}"


def build_all(*, config: Config | None = None, embedder=None, vector_index=None, full: bool = False,
              graph: bool = True) -> BuildStats:
    """Загрузка -> фрагменты -> эмбеддинги -> Chroma + BM25; затем граф. Инкрементально, по хэшам файлов.

    Без аргументов работает по Config.from_env(). embedder и vector_index можно подставить (тесты).
    EmbeddingError (Ollama недоступна, модель не скачана) пробрасывается: обработанные файлы уже сохранены.
    """
    from ..retrieval.lexical import Lexical
    from .vectorstore import VectorIndex

    t0 = time.perf_counter()
    cfg = config or Config.from_env()
    root = Path(cfg.repo_root)
    state_file = Path(cfg.index_dir) / HASHES_FILE
    vi = vector_index or VectorIndex(Path(cfg.index_dir) / "chroma", embedder=embedder)
    model = getattr(vi.embedder, "model", type(vi.embedder).__name__)
    meta = {"embed_model": model, "repo_root": str(root.resolve()), "chunker_version": CHUNKER_VERSION}

    state = _load_state(state_file)
    stats = BuildStats()
    old: dict[str, str] = state.get("files", {})
    if state.get("meta") != meta or full:
        stats.full_rebuild = True
        for path in sorted({c.path for c in vi.get_all()}):
            vi.delete_by_path(path)
        old = {}
    files = {p.relative_to(root).as_posix(): p for p in iter_files(root)}
    stats.files = len(files)
    current = dict(old)

    for rel in sorted(set(old) - set(files)):
        vi.delete_by_path(rel)
        current.pop(rel)
        stats.removed_files += 1
    _save_state(state_file, {"meta": meta, "files": current})

    for rel, path in files.items():
        digest = file_sha256(path)
        if old.get(rel) == digest:
            stats.unchanged_files += 1
            continue
        chunks = chunk_file(path, root)
        vi.delete_by_path(rel)
        vi.add(chunks)
        current[rel] = digest
        _save_state(state_file, {"meta": meta, "files": current})  # прерванная сборка продолжится с места
        stats.indexed_files += 1
        stats.new_chunks += len(chunks)

    all_chunks = vi.get_all()
    Lexical(all_chunks).save(Path(cfg.index_dir) / "bm25.json")
    stats.chunks = len(all_chunks)
    if graph:
        stats.graph = _build_graph(root, cfg)
    stats.seconds = time.perf_counter() - t0
    return stats
