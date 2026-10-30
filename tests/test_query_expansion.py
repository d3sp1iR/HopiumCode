"""retrieval.query_expansion: русские основы -> английские термины для поиска по коду."""
from hopiumcode.retrieval import query_expansion
from hopiumcode.retrieval.query_expansion import expand_query


def test_adds_english_terms_for_russian_stems():
    out = expand_query("Где создаётся сессия пользователя?")
    assert out.startswith("Где создаётся сессия пользователя?")
    assert "session" in out.split() and "create" in out.split()


def test_exact_keys_do_not_match_longer_words():
    assert "form" not in expand_query("В каком формате ответ?").split()
    assert "form" in expand_query("данные формы").split()


def test_no_terms_and_no_duplicates():
    assert expand_query("Flask make_response") == "Flask make_response"
    words = expand_query("ошибка и исключение в запросе").split()
    assert words.count("error") == 1 and words.count("exception") == 1 and "request" in words


def test_can_be_disabled(monkeypatch):
    monkeypatch.setenv("CG_QUERY_EXPANSION", "0")
    assert not query_expansion.enabled()
    monkeypatch.delenv("CG_QUERY_EXPANSION")
    assert query_expansion.enabled()
