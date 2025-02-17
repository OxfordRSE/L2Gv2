test: install-deps venv
	. .venv/bin/activate && python -m pytest -n auto
	
install-deps: venv
	. .venv/bin/activate && pip install '.[dev]'

venv:
	test -d .venv || python3.10 -m venv .venv
	 
.PHONY: test venv install-deps
