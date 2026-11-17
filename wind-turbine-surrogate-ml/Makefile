.PHONY: install test lint demo pipeline

install:
	pip install -e ".[dev]"

test:
	python -m pytest

lint:
	ruff check src tests examples

demo:
	python examples/demo.py

pipeline:
	python -m wtsurrogate.cli pipeline --out docs
