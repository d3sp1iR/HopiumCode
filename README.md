# hopiumcode: локальный ИИ-агент для онбординга в кодовую базу

> Рабочее название пакета `hopiumcode`. Команда CopiumCore. Переименование проекта: `scripts/rename_project.sh НОВОЕ_ИМЯ`.

Агент отвечает на вопросы разработчика по небольшому открытому репозиторию (до 100 файлов) и к каждому утверждению
даёт ссылку `[путь:начало-конец]`. Перед выдачей ответа ссылки проверяются: ссылка считается подтверждённой,
если её диапазон строк пересекается с найденным контекстом или с результатом инструмента. Всё работает
локально (Ollama, Chroma, граф кода, MCP-сервер), внешние API не используются.

Репозиторий-цель: Flask 3.1.3 (коммит `22d924701a6ae2e4cd01e9a15bbaf3946094af65`), индексируется только `src/flask`.

Архитектура: [docs/architecture.md](docs/architecture.md), схемы [docs/architecture.mmd](docs/architecture.mmd)
(Mermaid) и [docs/architecture.puml](docs/architecture.puml) (PlantUML). Зоны участников: [OWNERS.md](OWNERS.md).
Контракты: [docs/contracts.md](docs/contracts.md).

## Статус

Оркестратор, CLI, проверка ссылок, оценка и тесты работают. Индексация, ретривер, граф и MCP-сервер
(участники 2–4) пока заглушки (`NotImplementedError`). Цепочка вместо них подставляет заглушки из
`src/hopiumcode/stubs.py` и печатает предупреждение. Когда модуль перестаёт бросать `NotImplementedError`,
он подхватывается сам, менять код интеграции не нужно.

| Шаг | Что работает сейчас | Заглушка, пока модуль не готов |
| --- | --- | --- |
| `make fetch` | полностью | нет |
| `make index` | проверка, что Flask склонирован | `indexing.build.build_all` (участники 2 и 3) → `stubs.stub_build_all`: индекс не строится, только считаются .py-файлы |
| `make ask` | Ollama, промпт, цикл инструментов, проверка ссылок | `retrieval.hybrid.retrieve` (участник 3) → `stubs.stub_retrieve` с демо-фрагментами не из Flask; `mcp_client.McpClient` (участник 4) → `stubs.StubTools` |
| `make ask ARGS=--fake` | цепочка целиком без Ollama | всё: `ScriptedChat`, `stub_retrieve`, `StubTools` |
| `make eval` | метрики, таблицы, JSON и Markdown в `eval/results/` | режим, который ретривер не реализует, помечается «недоступно (заглушка)» и не прогоняется |

## Требования

- Linux или Windows с WSL2 (проверялось на Ubuntu 24.04 в WSL2), Python 3.11 или новее, git, make.
- [Ollama](https://ollama.com/download) с моделями:
  - LLM: `qwen3:8b` (по умолчанию) или `qwen3:4b`. qwen3:4b прошла `scripts/check_tools.py` на 3 из 3,
    qwen3:8b пока не проверялась (см. [docs/models.md](docs/models.md));
  - эмбеддинги: `nomic-embed-text`.
- Python-зависимости из `pyproject.toml`: `ollama`, `chromadb`, `rank-bm25`, `networkx`, `mcp`, `pyyaml`;
  для разработки `pytest`, `ruff`; по желанию `kuzu` (граф) и `gradio` (интерфейс).

## Установка (WSL / Ubuntu)

```bash
sudo apt install -y python3-venv git make
git clone <URL этого репозитория> HopiumCode && cd HopiumCode
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
# Ollama: установить по инструкции https://ollama.com/download, затем
ollama pull qwen3:8b            # или qwen3:4b, тогда export CG_LLM_MODEL=qwen3:4b
ollama pull nomic-embed-text
```

Make-цели сами берут `.venv/bin/python`, активировать venv не нужно (другой интерпретатор: `make test PYTHON=python3`).
Настройки читаются из переменных окружения `CG_*`, список с примерами в `.env.example`. Файл `.env`
автоматически не загружается, поэтому перед `make` выполните `set -a; source .env; set +a`.

## Быстрый старт

```bash
make fetch                                   # клон Flask 3.1.3 в data/repo/flask
make index                                   # индекс по data/repo/flask/src/flask
make ask Q="Где создаётся сессия пользователя?"
make ask Q="..." ARGS="--mode vector"        # режимы: vector | vector+bm25 | hybrid; --fake без Ollama; -v журнал шагов
make eval ARGS="--fake --mode all"           # оценка по eval/questions.yaml без Ollama
make test-all                                # pytest и быстрый eval на заглушках
```

| Цель | Что делает |
| --- | --- |
| `make fetch` | `scripts/fetch_repo.sh`: полный клон pallets/flask на зафиксированном коммите |
| `make index` | `python -m hopiumcode.cli index`; без клона подсказывает «Сначала выполните: make fetch» |
| `make ask Q="..."` | `python -m hopiumcode.cli ask`, флаги через `ARGS` |
| `make eval` | `python -m eval.run_eval`, флаги через `ARGS` (см. «Оценка») |
| `make test` / `make test-all` | pytest / pytest и `make eval ARGS="--fake --limit 3"` |
| `make lint` | ruff |

## Пример диалога (демо на заглушках)

Реальный вывод `make ask Q="Где создаётся сессия пользователя?" ARGS="--fake -v"`. Модель, ретривер и
инструменты здесь заглушки, поэтому ответ ссылается на демо-фрагмент `app/db.py` из `stubs.py`, а не на
код Flask. Пример показывает формат ответа и проверку ссылок, а не качество.

```text
{"event": "retrieved", "qtype": "semantic", "mode": "hybrid", "hits": 3}
{"event": "answer", "steps": 1, "tools": [], "good": 1, "bad": 0, "latency_s": 0.0}
(заглушка) Ответ по найденному фрагменту [app/db.py:10-24].

Подтверждено ссылок: 1 из 1.

[шагов модели: 1; инструменты: нет; 0.0 c]
```

Без `--fake` ответ даёт локальная модель. Пока ретривер и MCP-клиент не готовы, в stderr выводится:

```text
⚠ retrieval.hybrid.retrieve ещё не реализован (участник 3), используется заглушка stubs.stub_retrieve
⚠ mcp_client.McpClient ещё не реализован (участник 4), используется заглушка stubs.StubTools
```

## Оценка

`eval/questions.yaml` содержит 32 вопроса: 10 semantic, 10 graph, 5 overview, 5 history и 2 ловушки (trap).
Эталонные файлы, символы, диапазоны строк и коммиты проверены по коду Flask; `tests/test_eval_questions.py`
проверяет, что они существуют.

```bash
make eval ARGS="--mode all --limit 5"         # три режима подряд на первых 5 вопросах
make eval ARGS="--only q01,q20 --judge"       # выбранные вопросы и LLM-судья
```

| Метрика | Как считается |
| --- | --- |
| `recall@5`, `recall@8` | доля `expected_files` среди файлов первых k хитов retrieve |
| `citations_verified` | подтверждено ссылок / всего ссылок в ответе |
| `answered` | trap: «не нашёл» без ссылок; остальные типы: хотя бы одна подтверждённая ссылка |
| время, шаги | `Answer.latency_s`, `Answer.steps` (вызовы модели) |
| судья (`--judge`) | 0/1/2 по эталону, та же локальная модель, промпт П5.3; по умолчанию выключен |

Результат сохраняется в `eval/results/<дата-время>.json` и `.md`. В Markdown-отчёт попадают таблица
«режим × тип вопроса», итог по режимам и по 5 самых слабых вопросов на режим. Модель вызывается с
температурой 0. С `--fake` всё работает без Ollama и без клона Flask, но цифры проверяют только цепочку.

## Тесты

`make test` проходит без Ollama и без клона Flask (проверки по коду Flask в этом случае пропускаются).
Контрактные тесты (`tests/test_tools_contract.py`, `tests/test_chunker_contract.py`) пропускаются с указанием
причины, пока модуль участника бросает `NotImplementedError`, и включаются сами, когда модуль появится.
`tests/test_e2e_mini_repo.py` проверяет путь «вопрос → retrieve → ответ → проверка ссылок» на мини-репозитории
`tests/fixtures/mini_repo`.

## Ограничения

- Пока индексация, ретривер, граф и MCP — заглушки, агент не отвечает по коду Flask: заглушка ретривера
  возвращает три демо-фрагмента не из Flask.
- Проверка ссылок формальная: она подтверждает, что диапазон строк пересекается с найденным контекстом,
  но не то, что утверждение рядом со ссылкой верно.
- Агент отвечает на один вопрос без памяти диалога. Не больше `CG_MAX_STEPS` (по умолчанию 4) раундов
  вызова инструментов, контекст до 6000 символов, одна попытка исправить неподтверждённые ссылки.
- `recall@k` считается по первому вызову retrieve и не учитывает то, что агент нашёл инструментами.
- LLM-судья — та же модель, что отвечает, поэтому его оценки вспомогательные.
- `.env` не загружается автоматически. Корень индекса задаётся `CG_REPO_ROOT`, по умолчанию
  `data/repo/flask/src/flask`; какие расширения файлов индексировать, решает загрузчик (участник 2).

## Структура

| Папка | Что внутри | Владелец |
| --- | --- | --- |
| `src/hopiumcode/` | Оркестратор, CLI, промпты, конфиг, контракты, заглушки | Участник 1 |
| `src/hopiumcode/indexing/` | Загрузка, разбиение, эмбеддинги, векторная БД | Участник 2 |
| `src/hopiumcode/retrieval/` | BM25, слияние, гибридный ретривер | 2 и 3 |
| `src/hopiumcode/graph/` | Граф кода: извлечение, хранилище, запросы | Участник 3 |
| `src/hopiumcode/mcp_server/` | MCP-сервер с инструментами | Участник 4 |
| `eval/`, `tests/`, `docs/` | Оценка, тесты, документация | Участник 5 |
| `data/` | Репозиторий-цель и индексы (не коммитятся) | все |

Шаблон итогового отчёта: [docs/report_template.md](docs/report_template.md).
