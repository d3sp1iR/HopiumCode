"""Полный процесс индексации репозитория"""
from __future__ import annotations

from ..config import Config
from ..graph.build import build_graph


def build_all(config: Config | None = None) -> None:
    """Запускает полную сборку индексов и графа."""
    cfg = config or Config.from_env()
    build_graph(cfg)