from pathlib import Path

import pytest

from llmops_gate.datasets import generate_tickets, read_jsonl
from llmops_gate.registry import Registry
from llmops_gate.training import save_model, train_intent_classifier

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
DATASET = BENCHMARKS / "intent_benchmark.jsonl"
GATES = BENCHMARKS / "gates.yaml"


def _train(directory: Path, name: str, train_size: int) -> Path:
    records = generate_tickets(train_size, seed=1, id_prefix="train")
    onnx_bytes, metadata = train_intent_classifier(records)
    return save_model(directory / name, onnx_bytes, metadata)


@pytest.fixture(scope="session")
def good_model(tmp_path_factory) -> Path:
    """Classifier trained on 2000 synthetic tickets."""
    return _train(tmp_path_factory.mktemp("good"), "intent.onnx", 2000)


@pytest.fixture(scope="session")
def weak_model(tmp_path_factory) -> Path:
    """Classifier trained on only 60 synthetic tickets (a regressed candidate)."""
    return _train(tmp_path_factory.mktemp("weak"), "intent.onnx", 60)


@pytest.fixture(scope="session")
def benchmark() -> list[dict]:
    return read_jsonl(DATASET)


@pytest.fixture
def registry(tmp_path) -> Registry:
    return Registry(tmp_path / "registry")
