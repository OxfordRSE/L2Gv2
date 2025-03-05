REPO = https://github.com/OxfordRSE/L2Gv2/blob/main/

test: .venv
	. .venv/bin/activate && python -m pytest -n auto
	
lint: .venv
	. .venv/bin/activate && pylint l2gv2/**/*.py && pylint tests/**/*.py

.venv:
	python3.10 -m venv .venv && . .venv/bin/activate && pip install '.[dev]'

format:
	ruff format l2gv2
	ruff format tests

ruff-checks:
	ruff check l2gv2
	ruff format --check l2gv2
	ruff check tests
	ruff format --check tests

notebooks:
	python3.10 -m venv .venv && \
		. .venv/bin/activate && \
		pip install jupyter && \
		for i in examples/*.ipynb; do jupyter execute $$i; done

vulture: .venv
	@. .venv/bin/activate && vulture | grep -v variable |  awk -F : '{print "- [ ] [`" $$1 ":" $$2 "`]($(REPO)" $$1 "#L" $$2 "):" $$3}'


.PHONY: test lint ruff-checks format notebooks vulture
