import pytest
from fastapi.testclient import TestClient

from agentic_orchestrator.api import create_app


@pytest.fixture
def client(orchestrator) -> TestClient:
    return TestClient(create_app(orchestrator))


def test_health_and_tool_catalog(client):
    assert client.get("/health").json()["status"] == "ok"
    assert {t["name"] for t in client.get("/tools").json()} >= {"calculator", "issue_refund"}


def test_run_completes_and_can_be_fetched(client):
    created = client.post("/runs", json={"goal": "What is 6 * 7?"})
    assert created.status_code == 201
    body = created.json()
    assert body["status"] == "completed" and "42" in body["final_answer"]
    assert client.get(f"/runs/{body['run_id']}").json()["executed_tools"] == ["calculator"]


def test_approval_flow_over_http(client, backend):
    run = client.post("/runs", json={"goal": "Refund order ORD-1002"}).json()
    assert run["status"] == "awaiting_approval"
    pending = client.get("/approvals").json()
    assert len(pending) == 1 and pending[0]["tool_name"] == "issue_refund"
    assert pending[0]["risk"]["decision"] == "require_approval"

    decision = {"approved": True, "reviewer": "alice"}
    done = client.post(f"/runs/{run['run_id']}/approve", json=decision)
    assert done.status_code == 200 and done.json()["status"] == "completed"
    assert backend.refunds[0]["amount"] == 480.0
    assert client.get("/approvals").json() == []
    assert client.post(f"/runs/{run['run_id']}/approve", json=decision).status_code == 409


def test_rejection_without_reason_is_a_conflict(client):
    run = client.post("/runs", json={"goal": "Delete the record of customer CUST-001"}).json()
    response = client.post(
        f"/runs/{run['run_id']}/approve", json={"approved": False, "reviewer": "bob"}
    )
    assert response.status_code == 409 and "reason" in response.json()["detail"]


def test_unknown_run_and_invalid_payload(client):
    assert client.get("/runs/run_missing").status_code == 404
    missing = client.post("/runs/run_missing/approve", json={"approved": True, "reviewer": "a"})
    assert missing.status_code == 404
    assert client.post("/runs", json={"goal": ""}).status_code == 422
    assert client.post("/runs", json={"goal": "valid goal", "mode": "swarm"}).status_code == 422


def test_audit_endpoints(client):
    run = client.post("/runs", json={"goal": "Look up order ORD-1001"}).json()
    records = client.get("/audit", params={"run_id": run["run_id"]}).json()
    assert records[0]["event"] == "run_started" and records[-1]["event"] == "run_completed"
    only_tools = client.get("/audit", params={"event": "tool_executed"}).json()
    assert all(r["event"] == "tool_executed" for r in only_tools) and only_tools
    verify = client.get("/audit/verify").json()
    assert verify["ok"] is True and verify["records"] == len(client.get("/audit").json())
