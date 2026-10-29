PYTHON ?= .venv/bin/python
# доп. флаги для ask и eval, например ARGS="--fake --mode vector"
ARGS ?=

.PHONY: fetch index ask test test-all eval lint
fetch:
	bash scripts/fetch_repo.sh
index:
	$(PYTHON) -m hopiumcode.cli index
ask:
	$(PYTHON) -m hopiumcode.cli ask "$(Q)" $(ARGS)
test:
	$(PYTHON) -m pytest -q
test-all: test
	$(MAKE) --no-print-directory eval ARGS="--fake --limit 3"
eval:
	$(PYTHON) -m eval.run_eval $(ARGS)
lint:
	$(PYTHON) -m ruff check src tests eval
