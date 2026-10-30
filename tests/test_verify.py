import pytest

from hopiumcode.orchestrator import classify, parse_citations, verify_citations

ALLOWED = [("app/db.py", 10, 24), ("docs/usage.md", 20, 30)]


@pytest.mark.parametrize("text,ok,good,bad", [
    ("См. [app/db.py:10-24].", True, 1, 0),
    ("Часть функции [app/db.py:12-15].", True, 1, 0),           # диапазон внутри
    ("Пересечение [app/db.py:20-40].", True, 1, 0),             # диапазоны пересекаются
    ("Одна строка [app/db.py:11].", True, 1, 0),
    ("Выдумано [app/other.py:1-5].", False, 0, 1),              # неверный путь
    ("Мимо [app/db.py:100-120].", False, 0, 1),                 # нет пересечения
    ("Путь с ./ [./app/db.py:10-12].", True, 1, 0),
    ("Две: [app/db.py:10-12] и [x.py:1-2].", False, 1, 1),
    ("Без ссылок вообще.", True, 0, 0),
    ("", True, 0, 0),
    ("Перевёрнутый [app/db.py:24-10].", True, 1, 0),
])
def test_verify(text, ok, good, bad):
    r = verify_citations(text, ALLOWED)
    assert (r.ok, len(r.good), len(r.bad)) == (ok, good, bad)


def test_parse_normalizes_backslashes():
    assert parse_citations(r"[app\db.py:1-2]")[0][0] == "app/db.py"


@pytest.mark.parametrize("q,t", [
    ("Что вызывает функцию parse_config?", "graph"),
    ("Что сломается, если поменять класс X?", "graph"),
    ("Кто последним менял settings.py?", "history"),
    ("Где обрабатывается ошибка подключения?", "semantic"),
    ("Как устроен модуль авторизации", "semantic"),
    ("привет", "other"),
    ("", "other"),
])
def test_classify(q, t):
    assert classify(q) == t
