"""retrieval.lexical: токенизация, BM25, сохранение, RRF-слияние."""
import pytest

from hopiumcode.models import Chunk, Hit
from hopiumcode.retrieval.lexical import Lexical, fuse, split_identifier, tokenize


def ch(cid, text, name=None, path="m.py"):
    return Chunk(cid, path, 1, 2, "function", name, text, "python")


CHUNKS = [
    ch("m.py::parse_config", "def parse_config(path):\n    return load(path)", "parse_config"),
    ch("m.py::render", "def render(template):\n    return template.format()", "render"),
    ch("m.py::SecureCookieSession", "class SecureCookieSession(dict):\n    modified = False", "SecureCookieSession"),
    ch("m.py::save", "def save(data):\n    write(data)", "save"),
]


@pytest.mark.parametrize("word, parts", [
    ("parse_config", ["parse", "config"]),
    ("SecureCookieSession", ["Secure", "Cookie", "Session"]),
    ("HTTPException", ["HTTP", "Exception"]),
    ("max_content_length", ["max", "content", "length"]),
    ("_lazy_sha1", ["lazy", "sha", "1"]),
])
def test_split_identifier(word, parts):
    assert split_identifier(word) == parts


def test_tokenize_keeps_identifier_and_parts():
    assert tokenize("parse_config") == ["parse_config", "parse", "config"]
    assert tokenize("SecureCookieSession") == ["securecookiesession", "secure", "cookie", "session"]


def test_tokenize_drops_short_and_stop_words():
    assert tokenize("Где и как работает x в Flask?") == ["работает", "flask"]
    assert tokenize("the self of a config") == ["config"]
    assert tokenize("Ёлка") == ["елка"]


def test_query_parse_config_finds_parse_config():
    hits = Lexical(CHUNKS).search("parse config", 3)
    assert hits[0].chunk.id == "m.py::parse_config"
    assert hits[0].score == 1.0 and hits[0].via == "bm25"
    assert all(0 < h.score <= 1 for h in hits)


def test_camel_case_query():
    assert Lexical(CHUNKS).search("cookie session", 1)[0].chunk.id == "m.py::SecureCookieSession"


def test_no_match_and_empty_index():
    assert Lexical(CHUNKS).search("неизвестное слово", 5) == []
    assert Lexical([]).search("parse", 5) == []
    assert Lexical(CHUNKS).search("и в на", 5) == []           # только стоп-слова


def test_save_and_load(tmp_path):
    file = Lexical(CHUNKS).save(tmp_path / "bm25.json")
    loaded = Lexical(path=file)
    assert loaded.chunks == CHUNKS
    assert loaded.search("parse config", 1)[0].chunk == CHUNKS[0]


def test_default_loads_from_config_index_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CG_INDEX_DIR", str(tmp_path))
    assert Lexical().chunks == []                              # индекса ещё нет: пусто, без ошибки
    Lexical(CHUNKS).save()
    assert Lexical().search("render template", 1)[0].chunk.id == "m.py::render"


def H(c, score, via):
    return Hit(c, score, via)


def test_fuse_lifts_chunk_found_by_both():
    a, b, c, d = CHUNKS
    vector = [H(b, 0.9, "vector"), H(c, 0.8, "vector"), H(a, 0.7, "vector")]
    bm25 = [H(d, 1.0, "bm25"), H(a, 0.9, "bm25")]
    fused = fuse([vector, bm25])
    assert fused[0].chunk.id == a.id and fused[0].via == "vector+bm25"
    assert fused[0].score == 1.0 and all(0 < h.score <= 1 for h in fused)
    assert {h.via for h in fused[1:]} == {"vector", "bm25"}


def test_fuse_top_and_rrf_formula():
    a, b, *_ = CHUNKS
    fused = fuse([[H(a, 1, "vector"), H(b, 1, "vector")]], k=60, top=1)
    assert [h.chunk.id for h in fused] == [a.id]
    two = fuse([[H(a, 1, "vector"), H(b, 1, "vector")]])
    assert two[1].score == pytest.approx((1 / 62) / (1 / 61))


def test_fuse_merges_existing_via_and_ignores_duplicates():
    a, b, *_ = CHUNKS
    fused = fuse([[H(a, 1, "vector+bm25"), H(a, 1, "vector")], [H(a, 1, "graph:calls"), H(b, 1, "graph:calls")]])
    assert fused[0].via == "vector+bm25+graph:calls"
    assert fused[0].score == 1.0 and fused[1].chunk.id == b.id


def test_fuse_empty():
    assert fuse([]) == [] and fuse([[], []]) == []
