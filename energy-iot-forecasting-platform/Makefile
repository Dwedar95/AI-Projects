.PHONY: install test lint demo serve

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .

demo:
	python examples/run_demo.py

serve:
	uvicorn energy_iot.api:demo_app --factory --port 8000
