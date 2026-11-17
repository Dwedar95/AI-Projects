.PHONY: install test lint demo ablation

install:
	pip install -e ".[dev]"

test:
	python -m pytest

lint:
	ruff check src tests examples

demo:
	python examples/demo.py

ablation:
	python -m anomalybags.cli ablation --out docs
