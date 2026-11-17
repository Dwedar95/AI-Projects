.PHONY: install test lint demo gate benchmark serve

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

demo:
	PYTHONPATH=src python examples/demo.py

gate:
	PYTHONPATH=src python examples/train_intent_model.py --out artifacts/candidate.onnx
	PYTHONPATH=src python -m llmops_gate.cli --registry .registry register model \
		--name intent-classifier --artifact artifacts/candidate.onnx
	PYTHONPATH=src python -m llmops_gate.cli --registry .registry evaluate model \
		--name intent-classifier --dataset benchmarks/intent_benchmark.jsonl \
		--gates benchmarks/gates.yaml --baseline-metrics benchmarks/baseline_metrics.json

benchmark:
	PYTHONPATH=src python -m llmops_gate.cli benchmark

serve:
	PYTHONPATH=src LLMOPS_REGISTRY=.registry uvicorn llmops_gate.service:app --port 8080
