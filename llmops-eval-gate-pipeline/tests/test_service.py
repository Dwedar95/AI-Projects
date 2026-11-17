import pytest
from fastapi.testclient import TestClient

from llmops_gate.cost import TokenBudget
from llmops_gate.registry import Stage
from llmops_gate.service import create_app

PASS = {"passed": True, "checks": []}


def _release(registry, artifact):
    entry = registry.register_model("intent-classifier", artifact)
    registry.record_evaluation(entry, {"accuracy": 0.86}, PASS)
    registry.promote("model", "intent-classifier", entry.version, Stage.STAGING)
    return registry.promote("model", "intent-classifier", entry.version, Stage.PRODUCTION)


@pytest.fixture
def client(registry):
    budgets = {
        "default": TokenBudget(monthly_tokens=100_000, monthly_usd=5.0),
        "tiny": TokenBudget(monthly_tokens=12, monthly_usd=5.0),
    }
    return TestClient(create_app(registry, budgets=budgets))


def test_not_ready_without_a_production_model(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").status_code == 503
    assert client.post("/predict", json={"text": "hello"}).status_code == 503


def test_predict_serves_the_production_model_and_meters_usage(client, registry, good_model):
    _release(registry, good_model)
    assert client.get("/readyz").json()["model_ref"] == "model:intent-classifier:v1"
    response = client.post("/predict", json={"text": "my parcel tracking shows delivery delayed"})
    body = response.json()
    assert response.status_code == 200 and body["label"] == "shipping"
    assert body["model_ref"] == "model:intent-classifier:v1" and body["latency_ms"] > 0
    metrics = client.get("/metrics").json()
    assert metrics["by_team"][0]["team"] == "default" and metrics["by_team"][0]["requests"] == 1
    assert metrics["by_model"][0]["model"] == "onnx-classifier"


def test_budget_exhaustion_returns_429(client, registry, good_model):
    _release(registry, good_model)
    text = "password reset email not arriving"  # 8 estimated tokens
    assert client.post("/predict", json={"text": text, "team": "tiny"}).status_code == 200
    denied = client.post("/predict", json={"text": text, "team": "tiny"})
    assert denied.status_code == 429 and "budget" in denied.json()["detail"]
    assert client.post("/predict", json={"text": text, "team": "nobody"}).status_code == 429


def test_new_production_version_is_picked_up_without_restart(
    client, registry, good_model, weak_model
):
    _release(registry, good_model)
    assert client.post("/predict", json={"text": "invoice"}).json()["model_ref"].endswith("v1")
    _release(registry, weak_model)
    assert client.post("/predict", json={"text": "invoice"}).json()["model_ref"].endswith("v2")


def test_tampered_artifact_is_refused(client, registry, good_model):
    entry = _release(registry, good_model)
    registry.artifact_path(entry).write_bytes(b"corrupted")
    assert client.get("/readyz").status_code == 503
    assert "hash mismatch" in client.post("/predict", json={"text": "x"}).json()["detail"]


def test_request_validation(client):
    assert client.post("/predict", json={"text": ""}).status_code == 422
