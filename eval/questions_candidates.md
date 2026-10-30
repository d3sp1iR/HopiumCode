# Кандидаты в вопросы для добавления вопросов в набор до 30 (участник 5)

Сейчас в `questions.yaml` 13 вопросов (semantic 4, graph 3, history 2, overview 2, trap 2). По ТЗ нужно
10 semantic + 10 graph (связи) + 5 overview (объяснение модуля) + 5 history = 30.
Не хватает: **semantic +6, graph +7, overview +3, history +3** (и 2 trap остаются сверх).

Ниже только **формулировки и гипотезы, где искать**. Эталон (`expected_files`, `expected_symbols`,
`expected_ranges`, `reference_answer`) должен быть проверен вручную по клону `data/repo/flask` на
закреплённом коммите: читать код, `grep`, `git log`, `git blame`. Ничего отсюда нельзя копировать в
`questions.yaml` без этой проверки. В вопросах нет имён ответов (имена в графовых вопросах — это
аргумент вопроса, не ответ).

| Тип | Вопрос | Гипотеза, где искать (ПРОВЕРИТЬ) | Как проверить |
|---|---|---|---|
| semantic | Как Flask читает настройки из переменных окружения и из файла? | `src/flask/config.py` | прочитать класс Config, `grep -n "def from_" src/flask/config.py` |
| semantic | Где создаётся окружение шаблонизатора и как оно получает контекст приложения? | `templating.py`, `app.py` | `grep -n "jinja" src/flask/app.py` |
| semantic | Как Flask отдаёт статические файлы? | `app.py`, `helpers.py` | `grep -n "static" src/flask/app.py` |
| semantic | Как устроены «мгновенные» сообщения пользователю между запросами? | `helpers.py`, `sessions.py` | `grep -n "flash" src/flask/*.py` |
| semantic | Как объекты Python превращаются в JSON-ответ? | `json/` | прочитать `src/flask/json/*.py` |
| semantic | Как тестовый клиент строит запрос к приложению? | `testing.py` | прочитать `testing.py` |
| graph | Кто вызывает Flask.make_response? | `app.py` | `grep -rn "make_response" src/flask` |
| graph | Что вызывает Flask.dispatch_request? | `app.py` | прочитать тело метода |
| graph | Какие классы наследуются от SessionInterface? | `sessions.py` | `grep -rn "SessionInterface)" src/flask` |
| graph | Кто вызывает RequestContext.push? | `app.py`, `ctx.py`, `testing.py` | `grep -rn "\.push()" src/flask` |
| graph | Что сломается, если изменить сигнатуру Flask.process_response? | `app.py` | `grep -rn "process_response" src/flask` |
| graph | Где используется функция abort? | `helpers.py` и др. | `grep -rn "abort(" src/flask` |
| graph | Какие функции пакета вызывают url_for? | `helpers.py`, `app.py` | `grep -rn "url_for" src/flask` |
| overview | Как устроен подпакет flask.json? | `json/` | прочитать `json/__init__.py`, `provider.py`, `tag.py` |
| overview | За что отвечает модуль flask.cli? | `cli.py` | прочитать докстринги верхнего уровня |
| overview | Как устроен модуль flask.testing? | `testing.py` | прочитать классы модуля |
| history | Кто последним менял src/flask/sessions.py? | `git log -1` | `git log -1 --format=... -- src/flask/sessions.py` |
| history | Какой последний коммит менял src/flask/helpers.py? | `git log -1` | `git log -1 -- src/flask/helpers.py` |
| history | Когда и кем создан файл src/flask/sansio/scaffold.py? | `git log --diff-filter=A` | `git log --diff-filter=A -- src/flask/sansio/scaffold.py` |

Если вопрос не удаётся подтвердить по коду за 10 минут, замените его другим: лучше 27 проверенных вопросов
с честным пояснением в отчёте, чем 30 с непроверенным эталоном.
