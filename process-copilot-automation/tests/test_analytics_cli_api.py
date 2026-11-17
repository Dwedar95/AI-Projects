import pytest
from fastapi.testclient import TestClient

from process_copilot.analytics import (
    AdoptionMetrics,
    CaseRecord,
    case_log,
    comparison_table,
    compute_metrics,
    recommend,
    roi_projection,
)
from process_copilot.api import create_app
from process_copilot.cli import main
from process_copilot.process import load_process
from process_copilot.simulation import generate_cases, run_simulation
from process_copilot.tools import build_default_registry

from .conftest import PROCESSES


def rec(i, status="completed", automated=True, interventions=0, approvals=0, before=10.0, after=0.0, cost=0.001):
    return CaseRecord(f"c{i}", status, automated, interventions, approvals, before, after, cost, 100, 500.0, False,
                      "step_x" if status == "handoff" else "")


def test_metrics_hand_computed():
    records = [rec(1), rec(2), rec(3, automated=False, interventions=1, approvals=1, after=2.0),
               rec(4, status="handoff", automated=False, interventions=1, after=10.0)]
    m = compute_metrics(records, hourly_rate=60.0)
    assert m.automation_rate == 0.5 and m.intervention_rate == 0.5 and m.handoff_rate == 0.25
    assert m.approval_rate == 0.25 and m.interventions_per_case == 0.5
    assert m.avg_minutes_before == 10.0 and m.avg_minutes_after == 3.0
    assert m.handling_time_saved_pct == pytest.approx(0.7)
    assert m.cost_per_case_before == pytest.approx(10.0)  # 10 min at 60 USD/h
    assert m.cost_per_case_after == pytest.approx(3.0 + 0.001)
    assert m.saving_per_case == pytest.approx(6.999)
    assert m.handoff_steps == {"step_x": 1}
    assert "before (manual)" in comparison_table(m)
    with pytest.raises(ValueError):
        compute_metrics([])


def test_roi_projection_and_no_payback():
    m = compute_metrics([rec(1), rec(2, automated=False, interventions=1, after=4.0)], hourly_rate=60.0)
    roi = roi_projection(m, cases_per_month=1000, platform_cost_per_month=500.0, one_off_cost=15000.0)
    assert roi.gross_saving_per_month == pytest.approx((10.0 - 2.0 - 0.001) * 1000)
    assert roi.payback_months == pytest.approx(15000.0 / roi.net_saving_per_month)
    assert roi.hours_freed_per_month == pytest.approx(8.0 * 1000 / 60)
    assert roi_projection(m, 10, 500.0, 15000.0).payback_months is None


def _metrics(**kw) -> AdoptionMetrics:
    base = dict(cases=100, automation_rate=0.5, intervention_rate=0.5, interventions_per_case=0.5, handoff_rate=0.05,
                approval_rate=0.1, avg_minutes_before=10.0, avg_minutes_after=4.0, handling_time_saved_pct=0.6,
                cost_per_case_before=7.5, cost_per_case_after=3.0, model_cost_per_case=0.001, avg_tokens=200,
                avg_model_latency_ms=900, sla_breach_rate=0.0, hourly_rate=45.0)
    base.update(kw)
    return AdoptionMetrics(**base)


def test_playbook_recommendations():
    assert recommend(_metrics()).strategy == "augment"
    assert recommend(_metrics(automation_rate=0.8)).strategy == "rewrite"
    assert recommend(_metrics(automation_rate=0.8, handoff_rate=0.3, handoff_steps={"classify": 30})).strategy == "stabilise"
    assert recommend(_metrics(handling_time_saved_pct=0.1)).strategy == "stop"
    notes = recommend(_metrics(approval_rate=0.4, sla_breach_rate=0.2)).reasons
    assert any("approval thresholds" in n for n in notes) and any("SLA breach" in n for n in notes)


def test_simulation_is_reproducible_and_paired():
    process = load_process(PROCESSES / "hr_onboarding.yaml")
    a = compute_metrics(case_log(run_simulation(process, n=60, seed=3), seed=3))
    b = compute_metrics(case_log(run_simulation(process, n=60, seed=3), seed=3))
    c = compute_metrics(case_log(run_simulation(process, n=60, seed=4), seed=4))
    assert a == b and a != c
    assert 0 < a.automation_rate < 1 and a.avg_minutes_after < a.avg_minutes_before
    assert a.automation_rate + a.intervention_rate == pytest.approx(1.0)
    log = case_log(run_simulation(process, n=20, seed=3), seed=3, noise_sigma=0.0)
    assert all(r.human_minutes <= r.baseline_minutes + 1e-9 for r in log)


def test_case_generators():
    _, systems = build_default_registry()
    cases = generate_cases("customer_refund", 5, systems, seed=1)
    assert cases == generate_cases("customer_refund", 5, systems, seed=1)
    assert all({"customer_id", "message"} == set(c) for c in cases)
    with pytest.raises(ValueError):
        generate_cases("unknown", 1, systems)


def test_cli_commands(capsys):
    path = str(PROCESSES / "customer_refund.yaml")
    assert main(["validate", path]) == 0
    assert "OK: customer_refund" in capsys.readouterr().out
    assert main(["run", path, "--input", "customer_id=C-1001", "--input", "message=Please refund order ORD-1003"]) == 0
    assert "status: COMPLETED" in capsys.readouterr().out
    assert main(["run", path, "--input", "customer_id=C-1001"]) == 2
    assert main(["simulate", path, "--cases", "40"]) == 0
    out = capsys.readouterr().out
    assert "SIMULATED case log: 40 cases" in out and "Playbook recommendation" in out
    assert main(["tools"]) == 0
    assert "finance.issue_refund" in capsys.readouterr().out


def test_api_endpoints():
    client = TestClient(create_app())
    assert {p["id"] for p in client.get("/processes").json()} == {"customer_refund", "invoice_exception", "hr_onboarding"}
    assert any(t["name"] == "finance.issue_refund" and t["side_effect"] for t in client.get("/tools").json())
    body = {"inputs": {"employee_id": "E-3001", "question": "When is the payroll date?"}, "case_id": "api-1"}
    resp = client.post("/processes/hr_onboarding/run", json=body)
    assert resp.status_code == 200 and resp.json()["status"] == "completed" and resp.json()["case_id"] == "api-1"
    assert client.post("/processes/nope/run", json=body).status_code == 404
    assert client.post("/processes/hr_onboarding/run", json={"inputs": {}}).status_code == 422
