"""Обход репозитория. Владелец: участник 2."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

EXTENSIONS = {".py"}  # решение команды: Markdown и rst в индекс не берём (стабильность eval)
SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", "build", "dist"}
MAX_BYTES = 200_000
_SNIFF_BYTES = 8192


def is_binary(path: Path) -> bool:
    """Бинарный файл: нулевой байт в начале или текст не декодируется как UTF-8."""
    with path.open("rb") as f:
        head = f.read(_SNIFF_BYTES)
    if b"\0" in head:
        return True
    try:
        head.decode("utf-8")
    except UnicodeDecodeError as ex:
        # обрыв многобайтового символа на границе прочитанного блока не в счёт
        return not (len(head) == _SNIFF_BYTES and ex.start >= len(head) - 3)
    return False


def iter_files(root: Path) -> Iterator[Path]:
    """Файлы репозитория для индексации (без служебных каталогов, бинарных и больших файлов).

    Порядок стабильный: пути отсортированы по относительному пути с "/" в разделителях.
    """
    root = Path(root)
    found = []
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts[:-1]):
            continue
        if path.suffix not in EXTENSIONS or not path.is_file() or path.is_symlink():
            continue
        if path.stat().st_size > MAX_BYTES or is_binary(path):
            continue
        found.append((rel.as_posix(), path))
    for _, path in sorted(found):
        yield path
