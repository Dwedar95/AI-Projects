.PHONY: install test lint demo

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .

demo:
	python examples/run_demo.py
