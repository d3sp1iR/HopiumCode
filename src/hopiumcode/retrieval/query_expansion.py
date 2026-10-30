"""Расширение русского вопроса английскими терминами для поиска по коду.

Вопросы задаются по-русски, а код и докстринги Flask английские: BM25 без расширения не находит ни одного
совпадения, а эмбеддинги промахиваются (см. docs/indexing_report.md). Словарь общий для веб-разработки и
Python, он не подогнан под ответы eval. Ключ — начало русского слова (основа) или слово целиком, если
оканчивается на «!»; значение — английские термины.
Отключается переменной окружения CG_QUERY_EXPANSION=0.
"""
from __future__ import annotations

import os
import re

GLOSSARY: dict[str, str] = {
    "сесси": "session",
    "куки": "cookie",
    "кук": "cookie",
    "маршрут": "route url rule",
    "url": "url route",
    "адрес": "url",
    "запрос": "request",
    "ответ": "response",
    "обработ": "handle handler",
    "исключени": "exception error",
    "ошибк": "error exception",
    "конфигурац": "config",
    "настройк": "config settings",
    "шаблон": "template render",
    "контекст": "context",
    "сигнал": "signal",
    "команд": "command cli",
    "командн": "command cli",
    "консол": "cli shell",
    "тест": "test client",
    "блюпринт": "blueprint",
    "приложени": "app application",
    "расширени": "extension",
    "вызыва": "call",
    "наследу": "inherit subclass base class",
    "класс": "class",
    "функци": "function",
    "метод": "method",
    "модул": "module",
    "пакет": "package",
    "импорт": "import",
    "экспорт": "export __all__ import",
    "сериализ": "serialize json dumps",
    "json": "json",
    "подпис": "sign signature secret key",
    "ключ": "key",
    "секрет": "secret key",
    "заголов": "header headers",
    "статус": "status code",
    "перенаправ": "redirect",
    "файл": "file",
    "статическ": "static",
    "загруз": "load",
    "сохран": "save",
    "созда": "create open new",
    "откры": "open",
    "закры": "close teardown",
    "форма!": "form",
    "формы!": "form",
    "форме!": "form",
    "форму!": "form",
    "лимит": "limit max",
    "ограничен": "limit max",
    "размер": "size length",
    "хост": "host",
    "журнал": "log logger",
    "логир": "log logger",
    "отлад": "debug",
    "асинхрон": "async await",
    "декоратор": "decorator",
    "представлен": "view",
    "view": "view",
    "регистр": "register",
    "кодировк": "encoding",
    "дата": "date datetime",
    "даты": "date datetime",
    "хук": "hook before_request after_request",
}

_WORD = re.compile(r"[a-zа-яё_]+", re.IGNORECASE)


def enabled() -> bool:
    return os.environ.get("CG_QUERY_EXPANSION", "1") != "0"


def expand_query(question: str) -> str:
    """Вопрос плюс английские термины для русских слов, у которых есть основа в словаре."""
    extra: list[str] = []
    for word in _WORD.findall(question.lower().replace("ё", "е")):
        for stem, terms in GLOSSARY.items():
            if word == stem[:-1] if stem.endswith("!") else word.startswith(stem):
                extra.extend(t for t in terms.split() if t not in extra)
    return f"{question} {' '.join(extra)}" if extra else question
