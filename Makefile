PYTHON ?= .venv/bin/python
# доп. флаги для index, ask и eval, например ARGS="--fake --mode vector" или ARGS=--full
ARGS ?=

.PHONY: fetch index ask test eval lint
fetch:
	bash scripts/fetch_repo.sh
index:
	$(PYTHON) -m hopiumcode.indexing build $(ARGS)
ask:
	$(PYTHON) -m hopiumcode.cli ask "$(Q)" $(ARGS)
test:
	$(PYTHON) -m pytest -q
eval:
	$(PYTHON) -m eval.run_eval $(ARGS)
lint:
	$(PYTHON) -m ruff check src tests eval
