VENV?=.venv
PY?=$(VENV)/bin/python
PIP?=$(VENV)/bin/pip

.PHONY: venv install precommit fmt lint nb clean

venv:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

install: venv
	$(PIP) install -e .
	$(PIP) install black isort flake8 pre-commit

precommit:
	pre-commit install

fmt:
	black .
	isort .

lint:
	flake8 .

nb:
	$(VENV)/bin/jupyter lab

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name "*.pyc" -delete
