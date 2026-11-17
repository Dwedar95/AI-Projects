.PHONY: install test lint demo eval serve

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

demo:
	PYTHONPATH=src python examples/demo.py

eval:
	PYTHONPATH=src python -m agentic_orchestrator.evals --suite evals/support_ops.jsonl --baseline evals/baseline.json

serve:
	PYTHONPATH=src uvicorn agentic_orchestrator.api:app --port 8000
