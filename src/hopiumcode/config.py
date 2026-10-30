"""Конфигурация. Владелец: участник 1."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    llm_model: str = "qwen3:4b"  # проверена check_tools (docs/models.md); qwen3:8b через CG_LLM_MODEL
    embed_model: str = "nomic-embed-text"
    ollama_url: str = "http://localhost:11434"
    repo_root: Path = Path("data/repo/flask/src/flask")  # индексируется только src/flask (make fetch)
    index_dir: Path = Path("data/index")
    graph_dir: Path = Path("data/graph")
    top_k: int = 8
    max_steps: int = 4
    graph_backend: str = "networkx"  # networkx | kuzu
    repo_name: str = "Flask"  # имя проекта для системного промпта
    think: bool = False  # режим размышлений qwen3: точнее не стал, а ответ в 2-4 раза дольше

    @classmethod
    def from_env(cls) -> Config:
        e = os.environ.get
        return cls(
            llm_model=e("CG_LLM_MODEL", cls.llm_model),
            embed_model=e("CG_EMBED_MODEL", cls.embed_model),
            ollama_url=e("CG_OLLAMA_URL", cls.ollama_url),
            repo_root=Path(e("CG_REPO_ROOT", str(cls.repo_root))),
            index_dir=Path(e("CG_INDEX_DIR", str(cls.index_dir))),
            graph_dir=Path(e("CG_GRAPH_DIR", str(cls.graph_dir))),
            top_k=int(e("CG_TOP_K", cls.top_k)),
            max_steps=int(e("CG_MAX_STEPS", cls.max_steps)),
            graph_backend=e("CG_GRAPH_BACKEND", cls.graph_backend),
            repo_name=e("CG_REPO_NAME", cls.repo_name),
            think=e("CG_THINK", "0") == "1",
        )
