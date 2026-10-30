"""Обход репозитория. Владелец: участник 2."""
from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

log = logging.getLogger(__name__)

EXTENSIONS = {".py", ".md", ".rst", ".txt"}
SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", "build", "dist"}
MAX_BYTES = 200_000


def _is_text(path: Path) -> bool:
    """Бинарные файлы: есть нулевой байт в начале или содержимое не декодируется как UTF-8."""
    try:
        head = path.read_bytes()[:8192]
    except OSError:
        return False
    if b"\x00" in head:
        return False
    try:
        head.decode("utf-8")
    except UnicodeDecodeError as exc:
        # обрезка могла попасть посреди многобайтового символа: это не признак бинарного файла
        if exc.start < len(head) - 4:
            return False
    return True


def iter_files(root: Path) -> Iterator[Path]:
    """Файлы репозитория для индексации (без служебных каталогов, бинарных и больших файлов).

    Порядок детерминирован (сортировка по пути): от него зависят номера разделов документации
    и воспроизводимость индекса. Символические ссылки, выходящие за root, пропускаются.
    """
    root = Path(root)
    real_root = root.resolve()
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in EXTENSIONS:
            continue
        rel = path.relative_to(root)
        parts = rel.parts[:-1]
        if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in parts):
            continue
        if not path.is_file():
            continue
        if path.is_symlink() and not path.resolve().is_relative_to(real_root):
            log.warning("символическая ссылка за пределами репозитория пропущена: %s", rel)
            continue
        if path.stat().st_size > MAX_BYTES:
            log.warning("файл слишком большой, пропущен: %s", rel)
            continue
        if not _is_text(path):
            log.warning("не текстовый файл, пропущен: %s", rel)
            continue
        yield path
