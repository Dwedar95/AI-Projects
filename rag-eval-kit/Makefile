.PHONY: install test lint demo eval serve docker

install:
	pip install -e ".[dev]"

test:
	python -m pytest

lint:
	ruff check .
	ruff format --check .

demo:
	python examples/demo.py

# Runs the evaluation on the bundled synthetic data and applies the gate.
eval:
	rag-eval-kit eval --out reports

# Needs RAG_API_KEYS, for example: RAG_API_KEYS=app:reader:$$(openssl rand -hex 24) make serve
serve:
	rag-eval-kit serve

docker:
	docker compose up --build
