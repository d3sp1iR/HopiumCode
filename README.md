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
make eval ARGS="--fake"                     # прогон eval/questions.yaml без Ollama
make test
```

Make-цели сами берут `.venv/bin/python`, активировать venv не нужно (другой интерпретатор: `make test PYTHON=python3`).
Настройки читаются из переменных окружения `CG_*` (список в `.env.example`). Файл `.env` автоматически
не загружается: `set -a; source .env; set +a` перед `make`.

| Шаг | Что работает сейчас | Заглушка, пока модуль не готов |
|---|---|---|
| `make fetch` | полностью | нет |
| `make index` | проверка, что Flask склонирован | `indexing.build.build_all` (участники 2 и 3) → `stubs.stub_build_all`: индекс не строится, только считаются .py-файлы |
| `make ask` | Ollama, промпт, цикл инструментов, проверка ссылок | `retrieval.hybrid.retrieve` (участник 3) → `stubs.stub_retrieve` с демо-фрагментами не из Flask; `mcp_client.McpClient` (участник 4) → `stubs.StubTools` |
| `make ask ARGS=--fake` | цепочка целиком без Ollama | всё: `ScriptedChat`, `stub_retrieve`, `StubTools` |
| `make eval` | метрики, таблица, JSON в `eval/results/` | те же, что у `make ask`; с `--fake` метрики проверяют цепочку, а не качество |

При откате в stderr печатается предупреждение вида `⚠ retrieval.hybrid.retrieve ещё не реализован (участник 3),
используется заглушка stubs.stub_retrieve`. Когда модуль участника перестаёт бросать `NotImplementedError`,
цепочка сама переходит на него, менять код интеграции не нужно.

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

Статус модулей: индексация, ретривер, граф и MCP пока заглушки (`NotImplementedError`) с указанием владельца; оркестратор, CLI и оценка работают (см. таблицу выше). Контракты зафиксированы в `src/hopiumcode/models.py` и `interfaces.py`.
