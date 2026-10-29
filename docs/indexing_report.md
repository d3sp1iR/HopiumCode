# Отчёт по индексации и поиску (участник 2)

Дата прогона: 2026-10-29. Flask 3.1.3, коммит `22d924701a6ae2e4cd01e9a15bbaf3946094af65`, корень индекса
`data/repo/flask/src/flask`, только `.py`. Эмбеддинги `nomic-embed-text` через локальную Ollama.
Ветка `feature/indexing`.

## Индекс

| Показатель | Значение |
| --- | --- |
| Файлов | 24 |
| Фрагментов | 487: module 41, function 81, class 68, method 297 |
| Размерность эмбеддинга | 768 |
| Размер `data/index` | 12 МБ (Chroma, BM25, кэш эмбеддингов, хэши файлов) |

| Запуск `make index` | Время |
| --- | --- |
| Самый первый, пустой индекс и пустой кэш, модель эмбеддингов загружается в Ollama с нуля | 35.9 c |
| С нуля без кэша, модель уже загружена в Ollama | 16.1 c |
| Полная пересборка при заполненном кэше эмбеддингов (смена `CHUNKER_VERSION`) | 5.0 c |
| Повторный, без изменений: время сборки / вместе со стартом Python и импортом Chroma | 1.1 c / 3.2 c |

Первые замеры (35.9 c и повторный 1.3 c) сделаны на версии разбиения, где код модуля шёл одним фрагментом
(473 фрагмента). Проверка качества показала, что такой фрагмент мог охватывать сотни строк с чужими
функциями внутри (`cli.py:528-1135`), из-за чего проверка ссылок принимала бы почти любую ссылку на файл.
Разбиение исправлено: каждый фрагмент — непрерывный кусок файла; цифры выше и ниже — после исправления.

## Метод проверки

- 10 вопросов из `eval/questions.yaml` ветки `feature/eval-docs` (в `dev` пока 13 вопросов): все 5 overview
  (q10, q11, q27, q28, q29) и первые 5 semantic (q01, q02, q03, q04, q14). Вопросы задаются как есть, по-русски.
- vector: `VectorIndex().search(вопрос, 20)`, bm25: `Lexical().search(вопрос, 20)`, fuse: `fuse([vector, bm25], top=5)`.
  Для vector и bm25 в таблицу берутся первые 5 из 20.
- Попадание: хотя бы один файл из `expected_files` среди файлов топ-5. Критерий по файлу мягкий: файл может
  совпасть фрагментом, который на вопрос не отвечает (см. разбор).

## Сводка

| id | тип | вопрос | ожидаемые файлы | vector | bm25 | fuse |
| --- | --- | --- | --- | --- | --- | --- |
| q01 | semantic | Как Flask обрабатывает исключение, которое возникло во view-функции во время запроса? | `app.py`, `sansio/app.py` | да | да | да |
| q02 | semantic | Как работает маршрутизация во Flask: как URL сопоставляется с view-функцией? | `sansio/scaffold.py`, `sansio/app.py`, `app.py`, `ctx.py` | да | да | да |
| q03 | semantic | Где и когда создаётся (открывается) сессия пользователя и где она сохраняется? | `ctx.py`, `sessions.py`, `app.py` | **нет** | **нет** | **нет** |
| q04 | semantic | Как выполняются функции before_request и after_request и в каком порядке? | `app.py`, `sansio/scaffold.py` | да | да | да |
| q14 | semantic | Как Flask строит URL по имени endpoint и что происходит, если URL построить не удалось? | `helpers.py`, `app.py`, `sansio/app.py` | да | да | да |
| q10 | overview | Как устроен пакет flask и что он экспортирует наружу? | `__init__.py`, `app.py`, `blueprints.py`, `globals.py` | да | да | да |
| q11 | overview | За что отвечает подпакет flask.sansio? | `sansio/scaffold.py`, `sansio/app.py`, `sansio/blueprints.py` | да | **нет** | да |
| q27 | overview | Какие контексты есть во Flask и за что отвечает каждый из них? | `ctx.py`, `globals.py` | **нет** | да | **нет** |
| q28 | overview | Какие встроенные команды командной строки есть у Flask и как подключаются команды расширений? | `cli.py` | да | да | да |
| q29 | overview | Из каких частей состоит поддержка JSON во Flask и как она используется для сессий? | `json/__init__.py`, `json/provider.py`, `json/tag.py`, `sessions.py` | да | да | да |
| | | **итого попаданий в топ-5** | | **8/10** | **8/10** | **8/10** |

## Разбор, включая провалы

- **q03 (сессия) — провал всех трёх способов.** Вопрос по-русски без идентификаторов: BM25 не нашёл ни
  одного совпадения (пустая выдача), vector вернул `json/tag.py` (сериализатор, который сессии используют, но
  ответ на вопрос в `ctx.py` и `sessions.py`). Русский вопрос против английского кода — главный разрыв.
- **q10 (структура пакета) — попадание формальное.** Засчитано только через `app.py:81-105` (docstring класса
  `Flask`); `__init__.py`, где и лежит ответ (реэкспорт), не попал в топ-5 ни одним способом.
- **q27 (контексты) — fuse хуже BM25.** BM25 поставил `ctx.py:287-307 RequestContext` на 5-е место, vector его не
  нашёл вовсе, и после слияния фрагмент выпал из топ-5. Остальные места у всех способов заняты общими фрагментами
  (`FlaskClient.__init__`, `Flask`, справка CLI).
- **q11 (sansio) — BM25 мимо.** Совпадение по слову «sansio» дали блоки импортов (`blueprints.py:1-15` и др.), а не
  классы подпакета; vector нашёл `sansio/app.py:59-83 App`, fuse удержал `sansio/...` за счёт vector.
- **q28 (CLI) — попадание через справку.** И vector, и BM25 нашли `cli.py:1118-1127` (объект `cli = FlaskGroup(...)`
  с текстом справки); фрагменты `FlaskGroup.__init__` и `_load_plugin_commands`, где регистрируются команды, в топ-5
  не попали.
- **Удачные:** q04 (before/after_request — названия в вопросе совпадают с идентификаторами), q14 (URL — `url_for`),
  q29 (JSON — `json/*`). Здесь все три способа выдают нужные фрагменты в первой пятёрке.
- **Общие «магниты».** Фрагменты `testing.py:125-133 FlaskClient.__init__`, `app.py:81-105 Flask`,
  `wrappers.py:18-47 Request` и `cli.py:1118-1127` попадают в топ-5 vector на многих несвязанных вопросах.

## Что можно улучшить (не проверялось)

- Мост между русским вопросом и английским кодом: перевод или расширение запроса английскими терминами
  до поиска (зона оркестратора или ретривера), русские комментарии к именам при индексации.
- Префиксы задач `search_query:` / `search_document:`, которые рекомендует автор `nomic-embed-text`; в текущей
  версии не используются, эффект не измерялся.
- Понижение веса «магнитов» и коротких фрагментов из одного заголовка класса.


## Топ-5 по каждому вопросу

Формат: `путь:строки` (путь относительно `src/flask`), имя фрагмента, score. Ожидаемые файлы выделены жирным.

### q01 (semantic): Как Flask обрабатывает исключение, которое возникло во view-функции во время запроса?

Ожидаемые файлы: `app.py`, `sansio/app.py`

| # | vector | bm25 | fuse |
| --- | --- | --- | --- |
| 1 | **`app.py:1207-1216`** `Flask.make_response` 0.81 | `views.py:121-135` `View.as_view` 1.00 | `helpers.py:146-178` `make_response` 1.00 |
| 2 | **`app.py:81-105`** `Flask` 0.81 | `views.py:85-122` `View.as_view` 0.93 | **`app.py:1207-1216`** `Flask.make_response` 0.99 |
| 3 | `wrappers.py:18-47` `Request` 0.80 | `sansio/scaffold.py:701-706` `_endpoint_from_view_func` 0.93 | `wrappers.py:18-47` `Request` 0.98 |
| 4 | `testing.py:271-273` `FlaskCliRunner.__init__` 0.80 | `views.py:16-56` `View` 0.89 | **`app.py:1129-1168`** `Flask.make_response` 0.94 |
| 5 | `cli.py:1118-1127` `module` 0.80 | **`sansio/app.py:650-661`** `App.add_url_rule` 0.86 | `views.py:121-135` `View.as_view` 0.56 |

### q02 (semantic): Как работает маршрутизация во Flask: как URL сопоставляется с view-функцией?

Ожидаемые файлы: `sansio/scaffold.py`, `sansio/app.py`, `app.py`, `ctx.py`

| # | vector | bm25 | fuse |
| --- | --- | --- | --- |
| 1 | `wrappers.py:18-47` `Request` 0.81 | `sansio/blueprints.py:412-441` `Blueprint.add_url_rule` 1.00 | `wrappers.py:18-47` `Request` 1.00 |
| 2 | **`app.py:1003-1034`** `Flask.url_for` 0.81 | `sansio/blueprints.py:87-116` `BlueprintSetupState.add_url_rule` 0.95 | `sansio/blueprints.py:412-441` `Blueprint.add_url_rule` 0.97 |
| 3 | **`app.py:1033-1064`** `Flask.url_for` 0.80 | **`sansio/app.py:650-661`** `App.add_url_rule` 0.93 | **`app.py:1003-1034`** `Flask.url_for` 0.93 |
| 4 | `helpers.py:146-178` `make_response` 0.80 | `views.py:16-56` `View` 0.91 | `views.py:55-76` `View` 0.88 |
| 5 | `sessions.py:303-321` `SecureCookieSessionInterface.get_signing_serializer` 0.80 | **`sansio/scaffold.py:435-457`** `Scaffold.endpoint` 0.90 | `sansio/blueprints.py:87-116` `BlueprintSetupState.add_url_rule` 0.52 |

### q03 (semantic): Где и когда создаётся (открывается) сессия пользователя и где она сохраняется?

Ожидаемые файлы: `ctx.py`, `sessions.py`, `app.py`

| # | vector | bm25 | fuse |
| --- | --- | --- | --- |
| 1 | `blueprints.py:18-18` `Blueprint` 0.75 | — | `blueprints.py:18-18` `Blueprint` 1.00 |
| 2 | `json/tag.py:249-254` `TaggedJSONSerializer.__init__` 0.75 | — | `json/tag.py:249-254` `TaggedJSONSerializer.__init__` 0.98 |
| 3 | `json/tag.py:219-247` `TaggedJSONSerializer` 0.75 | — | `json/tag.py:219-247` `TaggedJSONSerializer` 0.97 |
| 4 | `json/tag.py:60-67` `JSONTag` 0.75 | — | `json/tag.py:60-67` `JSONTag` 0.95 |
| 5 | `json/tag.py:205-207` `TagDateTime` 0.75 | — | `json/tag.py:205-207` `TagDateTime` 0.94 |

### q04 (semantic): Как выполняются функции before_request и after_request и в каком порядке?

Ожидаемые файлы: `app.py`, `sansio/scaffold.py`

| # | vector | bm25 | fuse |
| --- | --- | --- | --- |
| 1 | `sansio/blueprints.py:553-561` `Blueprint.before_app_request` 0.83 | **`sansio/scaffold.py:486-505`** `Scaffold.after_request` 1.00 | `sansio/blueprints.py:553-561` `Blueprint.before_app_request` 1.00 |
| 2 | `sansio/blueprints.py:563-571` `Blueprint.after_app_request` 0.82 | **`sansio/scaffold.py:125-154`** `Scaffold.__init__` 0.94 | `sansio/blueprints.py:563-571` `Blueprint.after_app_request` 1.00 |
| 3 | `ctx.py:117-149` `after_this_request` 0.81 | `sansio/blueprints.py:563-571` `Blueprint.after_app_request` 0.92 | **`sansio/scaffold.py:486-505`** `Scaffold.after_request` 0.99 |
| 4 | **`sansio/scaffold.py:459-484`** `Scaffold.before_request` 0.81 | `sansio/blueprints.py:553-561` `Blueprint.before_app_request` 0.90 | **`sansio/scaffold.py:459-484`** `Scaffold.before_request` 0.97 |
| 5 | **`sansio/scaffold.py:486-505`** `Scaffold.after_request` 0.81 | **`sansio/scaffold.py:459-484`** `Scaffold.before_request` 0.86 | `ctx.py:309-335` `RequestContext.__init__` 0.94 |

### q14 (semantic): Как Flask строит URL по имени endpoint и что происходит, если URL построить не удалось?

Ожидаемые файлы: `helpers.py`, `app.py`, `sansio/app.py`

| # | vector | bm25 | fuse |
| --- | --- | --- | --- |
| 1 | **`app.py:1003-1034`** `Flask.url_for` 0.83 | `wrappers.py:146-159` `Request.endpoint` 1.00 | **`helpers.py:195-228`** `url_for` 1.00 |
| 2 | **`app.py:1033-1064`** `Flask.url_for` 0.82 | **`sansio/app.py:911-930`** `App.inject_url_defaults` 0.98 | `wrappers.py:146-159` `Request.endpoint` 0.99 |
| 3 | **`helpers.py:195-228`** `url_for` 0.82 | **`app.py:1063-1107`** `Flask.url_for` 0.97 | **`app.py:1003-1034`** `Flask.url_for` 0.98 |
| 4 | `wrappers.py:18-47` `Request` 0.81 | **`helpers.py:195-228`** `url_for` 0.96 | **`app.py:1033-1064`** `Flask.url_for` 0.98 |
| 5 | `sansio/blueprints.py:412-441` `Blueprint.add_url_rule` 0.81 | `sansio/blueprints.py:412-441` `Blueprint.add_url_rule` 0.95 | `sansio/blueprints.py:412-441` `Blueprint.add_url_rule` 0.98 |

### q10 (overview): Как устроен пакет flask и что он экспортирует наружу?

Ожидаемые файлы: `__init__.py`, `app.py`, `blueprints.py`, `globals.py`

| # | vector | bm25 | fuse |
| --- | --- | --- | --- |
| 1 | `testing.py:125-133` `FlaskClient.__init__` 0.82 | `testing.py:265-269` `FlaskCliRunner` 1.00 | **`app.py:81-105`** `Flask` 1.00 |
| 2 | **`app.py:81-105`** `Flask` 0.81 | `cli.py:1118-1127` `module` 0.98 | `cli.py:1118-1127` `module` 1.00 |
| 3 | `testing.py:271-273` `FlaskCliRunner.__init__` 0.81 | **`app.py:727-742`** `Flask.test_cli_runner` 0.98 | `testing.py:265-269` `FlaskCliRunner` 0.97 |
| 4 | `cli.py:1118-1127` `module` 0.81 | **`app.py:81-105`** `Flask` 0.94 | `testing.py:271-273` `FlaskCliRunner.__init__` 0.96 |
| 5 | `testing.py:109-123` `FlaskClient` 0.80 | `ctx.py:287-307` `RequestContext` 0.94 | `cli.py:267-280` `get_version` 0.94 |

### q11 (overview): За что отвечает подпакет flask.sansio?

Ожидаемые файлы: `sansio/scaffold.py`, `sansio/app.py`, `sansio/blueprints.py`

| # | vector | bm25 | fuse |
| --- | --- | --- | --- |
| 1 | **`sansio/app.py:59-83`** `App` 0.83 | `blueprints.py:1-15` `module` 1.00 | `blueprints.py:18-18` `Blueprint` 1.00 |
| 2 | `cli.py:229-232` `locate_app` 0.81 | `json/provider.py:1-16` `module` 0.90 | **`sansio/blueprints.py:34-39`** `BlueprintSetupState` 1.00 |
| 3 | `cli.py:1118-1127` `module` 0.81 | `debughelpers.py:1-14` `module` 0.89 | **`sansio/app.py:479-480`** `App.create_jinja_environment` 0.93 |
| 4 | `app.py:81-105` `Flask` 0.81 | `templating.py:1-21` `module` 0.88 | **`sansio/app.py:52-56`** `_make_timedelta` 0.92 |
| 5 | `cli.py:235-238` `locate_app` 0.81 | `blueprints.py:18-18` `Blueprint` 0.85 | **`sansio/blueprints.py:119-133`** `Blueprint` 0.88 |

### q27 (overview): Какие контексты есть во Flask и за что отвечает каждый из них?

Ожидаемые файлы: `ctx.py`, `globals.py`

| # | vector | bm25 | fuse |
| --- | --- | --- | --- |
| 1 | `testing.py:125-133` `FlaskClient.__init__` 0.80 | `testing.py:265-269` `FlaskCliRunner` 1.00 | `cli.py:1118-1127` `module` 1.00 |
| 2 | `cli.py:1118-1127` `module` 0.80 | `cli.py:1118-1127` `module` 0.98 | `app.py:81-105` `Flask` 0.97 |
| 3 | `cli.py:267-280` `get_version` 0.79 | `app.py:727-742` `Flask.test_cli_runner` 0.98 | `cli.py:267-280` `get_version` 0.95 |
| 4 | `app.py:81-105` `Flask` 0.79 | `app.py:81-105` `Flask` 0.94 | `testing.py:271-273` `FlaskCliRunner.__init__` 0.93 |
| 5 | `testing.py:271-273` `FlaskCliRunner.__init__` 0.79 | **`ctx.py:287-307`** `RequestContext` 0.94 | `testing.py:109-123` `FlaskClient` 0.90 |

### q28 (overview): Какие встроенные команды командной строки есть у Flask и как подключаются команды расширений?

Ожидаемые файлы: `cli.py`

| # | vector | bm25 | fuse |
| --- | --- | --- | --- |
| 1 | `testing.py:125-133` `FlaskClient.__init__` 0.80 | `testing.py:265-269` `FlaskCliRunner` 1.00 | **`cli.py:1118-1127`** `module` 1.00 |
| 2 | **`cli.py:1118-1127`** `module` 0.80 | **`cli.py:1118-1127`** `module` 0.98 | `app.py:81-105` `Flask` 0.98 |
| 3 | `app.py:81-105` `Flask` 0.79 | `app.py:727-742` `Flask.test_cli_runner` 0.98 | `ctx.py:287-307` `RequestContext` 0.94 |
| 4 | `wrappers.py:222-244` `Response` 0.79 | `app.py:81-105` `Flask` 0.94 | `testing.py:271-273` `FlaskCliRunner.__init__` 0.93 |
| 5 | `testing.py:271-273` `FlaskCliRunner.__init__` 0.79 | `ctx.py:287-307` `RequestContext` 0.94 | **`cli.py:267-280`** `get_version` 0.91 |

### q29 (overview): Из каких частей состоит поддержка JSON во Flask и как она используется для сессий?

Ожидаемые файлы: `json/__init__.py`, `json/provider.py`, `json/tag.py`, `sessions.py`

| # | vector | bm25 | fuse |
| --- | --- | --- | --- |
| 1 | **`json/__init__.py:108-135`** `load` 0.82 | **`json/provider.py:19-36`** `JSONProvider` 1.00 | **`json/__init__.py:108-135`** `load` 1.00 |
| 2 | `wrappers.py:222-244` `Response` 0.82 | **`json/__init__.py:77-105`** `loads` 0.96 | **`json/__init__.py:77-105`** `loads` 0.99 |
| 3 | **`json/__init__.py:77-105`** `loads` 0.81 | **`json/__init__.py:108-135`** `load` 0.96 | **`json/provider.py:19-36`** `JSONProvider` 0.96 |
| 4 | **`json/provider.py:189-215`** `DefaultJSONProvider.response` 0.81 | **`json/__init__.py:47-74`** `dump` 0.96 | **`json/__init__.py:47-74`** `dump` 0.95 |
| 5 | **`json/__init__.py:138-170`** `jsonify` 0.81 | **`json/__init__.py:13-44`** `dumps` 0.95 | **`json/__init__.py:13-44`** `dumps` 0.95 |

