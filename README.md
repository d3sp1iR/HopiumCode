# hopiumcode: локальный ИИ-агент, помощник по кодовой базе

> Рабочее название пакета `hopiumcode`. Команда CopiumCore. Переименование проекта: `scripts/rename_project.sh НОВОЕ_ИМЯ`.

Агент отвечает на вопросы разработчика по небольшому открытому репозиторию (до 100 файлов) и в каждом утверждении
даёт ссылку `[путь:начало-конец]`. Всё работает локально: Ollama, Chroma, граф кода, MCP-сервер, без внешних API.

Архитектура и задания участников: см. `docs/architecture.md` и `OWNERS.md`. Полное ТЗ лежит в `docs/` (PDF).

## Быстрый старт

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"   # Windows: .venv\Scripts\pip
ollama pull qwen3:8b && ollama pull nomic-embed-text         # другая модель: CG_LLM_MODEL=qwen3:4b
make fetch                                  # клон Flask 3.1.3 в data/repo/flask
make index                                  # индекс по data/repo/flask/src/flask
make ask Q="Где создаётся сессия?"          # флаги: ARGS="--mode vector" или ARGS="--fake"
make eval ARGS="--fake"                     # прогон eval/questions.yaml без Ollama (проверка каркаса)
make test
```

Make-цели сами берут `.venv/bin/python`, активировать venv не нужно (другой интерпретатор: `make test PYTHON=python3`).
Настройки читаются из переменных окружения `CG_*` (список в `.env.example`). Файл `.env` автоматически
не загружается: `set -a; source .env; set +a` перед `make`.

| Шаг                    | Что делает                                                                                  |
|------------------------|---------------------------------------------------------------------------------------------|
| `make fetch`           | клонирует Flask на закреплённый коммит в `data/repo/flask`                                  |
| `make index`           | фрагменты → эмбеддинги Ollama → Chroma (инкрементально по sha256 файлов) + граф кода         |
| `make ask`             | гибридный поиск, цикл инструментов MCP, проверка ссылок (нужны Ollama и построенный индекс)   |
| `make ask ARGS=--fake` | цепочка целиком на заглушках, без Ollama (проверка каркаса)                                  |
| `make eval`            | метрики, таблица, JSON в `eval/results/`; с `--fake` метрики проверяют цепочку, не качество |

Повторный `make index` без изменений в репозитории пропускает разбиение и эмбеддинги (только пересобирает граф).
При смене `CG_EMBED_MODEL` индекс пересоздаётся целиком. Если индекс не построен, `make ask` подскажет
выполнить `make index`.

Проверка точности CALLS (критерий AC-4): `python scripts/check_calls.py sample` создаёт чек-лист на 30
рёбер в `docs/calls_check.md`, после ручной разметки `python scripts/check_calls.py score docs/calls_check.md`.

Оценка: `make eval ARGS="--mode vector+bm25 --only q01,q05"`. Метрики на вопрос: `hit_files` (доля
`expected_files` среди найденных фрагментов и подтверждённых ссылок), `citations_verified`, `answered`
(для `trap`: честное «не нашёл» без ссылок), время и число вызовов инструментов; итоги по типам вопросов.

## Структура

| Папка | Что внутри | Владелец |
|---|---|---|
| `src/hopiumcode/` | Код агента | см. `OWNERS.md` |
| `src/hopiumcode/indexing/` | Загрузка, разбиение, эмбеддинги, векторная БД | Участник 2 |
| `src/hopiumcode/retrieval/` | BM25, слияние, гибридный ретривер | 2 и 3 |
| `src/hopiumcode/graph/` | Граф кода: извлечение, хранилище, запросы | Участник 3 |
| `src/hopiumcode/mcp_server/` | MCP-сервер с инструментами | Участник 4 |
| `eval/`, `tests/`, `docs/` | Оценка, тесты, документация | Участник 5 |
| `data/` | Репозиторий-цель и индексы (не коммитятся) | все |

Статус модулей: индексация (`indexing/`, `retrieval/lexical.py`), граф (`graph/`), гибридный ретривер,
MCP-сервер и клиент, оркестратор, консоль и оценка реализованы. Не реализован `KuzuStore` (используется
NetworkX, см. `docs/decisions.md`). Контракты зафиксированы в `src/hopiumcode/models.py` и `interfaces.py`;
соглашения об идентификаторах фрагментов: `docs/contracts.md`. Что ещё нужно сделать до сдачи: `docs/TODO.md`.
