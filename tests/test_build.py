"""indexing.build.build_all и python -m hopiumcode.indexing: инкрементальная сборка на временном репозитории.

Эмбеддинги: HashEmbedder (без Ollama), Chroma во временной папке.
"""
import textwrap

import pytest

from hopiumcode.config import Config
from hopiumcode.indexing import __main__ as cmd
from hopiumcode.indexing.build import HASHES_FILE, build_all
from hopiumcode.indexing.embed import EmbeddingError, HashEmbedder
from hopiumcode.indexing.vectorstore import VectorIndex
from hopiumcode.retrieval.lexical import Lexical


class NamedHash(HashEmbedder):
    def __init__(self, model="hash-v1", fail_after=None):
        super().__init__(dim=64)
        self.model, self.fail_after, self.texts = model, fail_after, 0

    def embed(self, texts):
        self.texts += len(texts)
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise EmbeddingError("Ollama недоступна (тест)")
        return super().embed(texts)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "config.py").write_text("def parse_config(path):\n    return path\n")
    (root / "pkg" / "app.py").write_text(textwrap.dedent('''\
        import os


        class App:
            def run(self):
                return os.getcwd()
    '''))
    (root / "README.md").write_text("# не индексируется\n")
    return root


def run(repo, tmp_path, embedder=None, **kw):
    cfg = Config(repo_root=repo, index_dir=tmp_path / "index")
    emb = embedder or NamedHash()
    stats = build_all(config=cfg, embedder=emb, **kw)
    return stats, emb, cfg


def test_first_build_indexes_everything(repo, tmp_path):
    stats, emb, cfg = run(repo, tmp_path)
    assert (stats.files, stats.indexed_files, stats.unchanged_files) == (2, 2, 0)
    assert stats.chunks == stats.new_chunks > 0 and stats.full_rebuild
    assert (cfg.index_dir / HASHES_FILE).exists() and (cfg.index_dir / "bm25.json").exists()
    assert "не реализован" in stats.graph                     # graph.extract пока стаб (участник 3)
    bm25 = Lexical(path=cfg.index_dir / "bm25.json")
    assert bm25.search("parse config", 1)[0].chunk.id == "config.py::parse_config"
    vi = VectorIndex(cfg.index_dir / "chroma", embedder=emb)
    assert {c.path for c in vi.get_all()} == {"config.py", "pkg/app.py"}
    assert {c.id for c in vi.get_all()} == {c.id for c in bm25.chunks}


def test_second_run_skips_unchanged_without_embedding(repo, tmp_path):
    run(repo, tmp_path)
    stats, emb, _ = run(repo, tmp_path)
    assert (stats.unchanged_files, stats.indexed_files, stats.new_chunks) == (2, 0, 0)
    assert emb.calls == 0 and not stats.full_rebuild
    assert stats.chunks > 0


def test_changed_and_removed_files(repo, tmp_path):
    run(repo, tmp_path)
    (repo / "config.py").write_text("def load_settings(path):\n    return path\n")
    (repo / "pkg" / "app.py").unlink()
    stats, _, cfg = run(repo, tmp_path)
    assert (stats.indexed_files, stats.removed_files, stats.unchanged_files) == (1, 1, 0)
    names = {c.name for c in Lexical(path=cfg.index_dir / "bm25.json").chunks}
    assert "load_settings" in names and "parse_config" not in names and "App.run" not in names


def test_model_change_forces_full_rebuild(repo, tmp_path):
    run(repo, tmp_path)
    stats, _, _ = run(repo, tmp_path, embedder=NamedHash(model="hash-v2"))
    assert stats.full_rebuild and stats.indexed_files == 2


def test_interrupted_build_resumes(repo, tmp_path):
    with pytest.raises(EmbeddingError):
        run(repo, tmp_path, embedder=NamedHash(fail_after=1))   # первый файл успел, второй нет
    stats, _, _ = run(repo, tmp_path)
    assert (stats.unchanged_files, stats.indexed_files) == (1, 1)


def test_cli_without_sources_hints_fetch(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CG_REPO_ROOT", str(tmp_path / "nothing"))
    assert cmd.main(["build"]) == 1
    assert "make fetch" in capsys.readouterr().err


def test_cli_ollama_error_is_readable(repo, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CG_REPO_ROOT", str(repo))
    monkeypatch.setenv("CG_INDEX_DIR", str(tmp_path / "index"))

    def fail(**kw):
        raise EmbeddingError("Ollama недоступна (http://localhost:11434). Запустите `ollama serve` и повторите.")

    monkeypatch.setattr("hopiumcode.indexing.build.build_all", fail)
    assert cmd.main(["build"]) == 2
    err = capsys.readouterr().err
    assert "ollama serve" in err and "Traceback" not in err


def test_cli_prints_stats(repo, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CG_REPO_ROOT", str(repo))
    monkeypatch.setenv("CG_INDEX_DIR", str(tmp_path / "index"))
    real = build_all
    monkeypatch.setattr("hopiumcode.indexing.build.build_all", lambda **kw: real(embedder=NamedHash(), **kw))
    assert cmd.main(["build", "--no-graph"]) == 0
    assert "Файлов: 2" in capsys.readouterr().out
