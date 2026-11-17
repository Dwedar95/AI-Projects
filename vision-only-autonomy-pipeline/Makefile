.PHONY: install test lint demo

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src tests examples

demo:
	python examples/run_demo.py
