test: install-deps venv
	. .venv/bin/activate && python -m pytest -n auto
	
install-deps: venv
	. .venv/bin/activate && pip install '.[dev]'

venv:
	test -d .venv || python3.10 -m venv .venv

ruff-checks:
	ruff check l2gv2
	ruff format --check l2gv2
	ruff check tests
	ruff format --check tests

.PHONY: test venv install-deps
