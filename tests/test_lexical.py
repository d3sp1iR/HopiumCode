"""BM25, токенизация и слияние RRF (участник 2)."""
from hopiumcode.models import Chunk, Hit
from hopiumcode.retrieval.lexical import Lexical, _BM25, fuse, split_identifier, tokenize


def chunk(cid: str, text: str, name: str | None = None, path: str = "a.py") -> Chunk:
    return Chunk(cid, path, 1, 5, "function", name, text, "python")


def test_split_identifier():
    assert split_identifier("parse_config") == ["parse", "config"]
    assert split_identifier("parseConfig") == ["parse", "config"]
    assert split_identifier("HTTPError") == ["http", "error"]


def test_tokenize():
    toks = tokenize("Where is parse_config used? Как обрабатывается ошибка")
    assert "parse" in toks and "config" in toks and "parse_config" in toks
    assert "where" not in toks and "is" not in toks and "как" not in toks
    assert "ошибка" in toks and all(len(t) >= 2 for t in toks)


def test_bm25_finds_function_by_split_name():
    chunks = [chunk("a.py::parse_config", "def parse_config(path):\n    return load(path)", "parse_config"),
              chunk("a.py::other", "def other():\n    return 1", "other"),
              chunk("a.py::render", "def render(template):\n    return template", "render")]
    hits = Lexical(chunks).search("parse config", 3)
    assert hits and hits[0].chunk.id == "a.py::parse_config"
    assert hits[0].score == 1.0 and all(0 <= h.score <= 1 for h in hits) and hits[0].via == "bm25"


def test_bm25_empty_cases():
    assert Lexical([]).search("anything", 5) == []
    assert Lexical([chunk("x", "def a(): pass")]).search("   ", 5) == []
    assert Lexical([chunk("x", "def a(): pass")]).search("zzzzunknown", 5) == []


def test_fallback_bm25_matches_ranking():
    corpus = [tokenize(t) for t in ("alpha beta beta", "beta gamma", "delta epsilon")]
    scores = _BM25(corpus).get_scores(tokenize("beta"))
    assert scores[0] > scores[1] > 0 and scores[2] == 0


def test_fuse_boosts_chunk_found_by_both():
    a, b, c = (chunk(x, x) for x in "abc")
    vec = [Hit(a, 0.9, "vector"), Hit(b, 0.8, "vector")]
    bm = [Hit(c, 1.0, "bm25"), Hit(b, 0.5, "bm25")]
    out = fuse([vec, bm])
    assert out[0].chunk.id == "b" and out[0].via == "vector+bm25"
    assert {h.chunk.id for h in out} == {"a", "b", "c"}
    assert all(0 <= h.score <= 1 for h in out)


def test_fuse_top_and_empty():
    hits = [Hit(chunk(str(i), "x"), 1 - i / 10, "vector") for i in range(10)]
    assert len(fuse([hits], top=3)) == 3
    assert fuse([]) == [] and fuse([[], []]) == []
    dup = [Hit(hits[0].chunk, 0.5, "vector"), Hit(hits[0].chunk, 0.4, "vector")]
    assert len(fuse([dup])) == 1
