PYTHON ?= .venv/bin/python
# доп. флаги для ask и eval, например ARGS="--fake --mode vector"
ARGS ?=

.PHONY: fetch index ask test eval lint check-calls
fetch:
	bash scripts/fetch_repo.sh
index:
	$(PYTHON) -m hopiumcode.cli index
ask:
	$(PYTHON) -m hopiumcode.cli ask "$(Q)" $(ARGS)
test:
	$(PYTHON) -m pytest -q
eval:
	$(PYTHON) -m eval.run_eval $(ARGS)
lint:
	$(PYTHON) -m ruff check src tests eval
check-calls:
	$(PYTHON) scripts/check_calls.py sample
