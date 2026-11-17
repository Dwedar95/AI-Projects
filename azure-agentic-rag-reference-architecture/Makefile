.PHONY: install test lint demo validate bicep-build

install:
	pip install -e ".[dev]"

test:
	python -m pytest

lint:
	ruff check .
	ruff format --check .

demo:
	python examples/demo.py

validate:
	archkit validate-bicep infra

# Requires the Bicep CLI on PATH (https://aka.ms/bicep-install).
bicep-build:
	mkdir -p out
	bicep build infra/main.bicep --outdir out
	bicep build-params infra/parameters/dev.bicepparam --outdir out
	bicep build-params infra/parameters/prod.bicepparam --outdir out
