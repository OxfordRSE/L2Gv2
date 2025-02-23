test: .venv
	. .venv/bin/activate && python -m pytest -n auto
	
lint: .venv
	. .venv/bin/activate && pylint {l2gv2,tests}/**/*.py

.venv:
	python3.10 -m venv .venv && . .venv/bin/activate && pip install '.[dev]'

ruff-checks:
	ruff check l2gv2
	ruff format --check l2gv2
	ruff check tests
	ruff format --check tests

.PHONY: test lint ruff-checks
