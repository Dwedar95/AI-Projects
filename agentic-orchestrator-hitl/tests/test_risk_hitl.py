import pytest

from agentic_orchestrator.hitl import ApprovalError, ApprovalGate
from agentic_orchestrator.models import ApprovalStatus, RiskDecision
from agentic_orchestrator.risk import RiskPolicy, RiskScorer


@pytest.mark.parametrize(
    ("tool", "args", "tainted", "score", "decision"),
    [
        ("kb_search", {"query": "x"}, False, 0.0, RiskDecision.ALLOW),
        ("kb_search", {"query": "x"}, True, 0.0, RiskDecision.ALLOW),
        ("issue_refund", {"order_id": "ORD-1001", "amount": 40}, False, 0.35, RiskDecision.ALLOW),
        ("issue_refund", {"order_id": "ORD-1002", "amount": 480}, False, 0.55, "require_approval"),
        ("issue_refund", {"order_id": "ORD-1003", "amount": 2300}, False, 0.75, "require_approval"),
        ("issue_refund", {"order_id": "ORD-1001", "amount": 40}, True, 0.65, "require_approval"),
        ("send_email", {"to": "ops@company.example"}, False, 0.40, RiskDecision.ALLOW),
        ("send_email", {"to": "b@customer.example"}, False, 0.65, "require_approval"),
        ("send_email", {"to": "x@attacker.example"}, True, 0.95, RiskDecision.DENY),
        ("delete_customer_record", {"customer_id": "CUST-001"}, False, 0.80, "require_approval"),
        ("delete_customer_record", {"customer_id": "CUST-001"}, True, 1.0, RiskDecision.DENY),
    ],
)
def test_risk_scores_and_decisions(registry, tool, args, tainted, score, decision):
    assessment = RiskScorer().assess(registry.get(tool), args, tainted)
    assert assessment.score == pytest.approx(score)
    assert assessment.decision == decision


def test_reasons_explain_every_contribution(registry):
    reasons = RiskScorer().assess(registry.get("send_email"), {"to": "a@b.example"}, True).reasons
    joined = " | ".join(reasons)
    assert "external" in joined and "irreversible" in joined and "untrusted content" in joined


def test_policy_thresholds_are_validated():
    with pytest.raises(ValueError):
        RiskPolicy(approval_threshold=0.9, deny_threshold=0.5)


def test_stricter_policy_escalates_small_refund(registry):
    scorer = RiskScorer(RiskPolicy(approval_threshold=0.3))
    args = {"order_id": "ORD-1001", "amount": 40}
    decision = scorer.assess(registry.get("issue_refund"), args, False).decision
    assert decision is RiskDecision.REQUIRE_APPROVAL


def test_gate_creates_request_only_for_approval_band(registry):
    gate = ApprovalGate()
    low = gate.evaluate("run_1", 1, registry.get("calculator"), {"expression": "1+1"}, False)
    high = gate.evaluate(
        "run_1", 1, registry.get("delete_customer_record"), {"customer_id": "CUST-001"}, False
    )
    assert low.request is None and low.decision is RiskDecision.ALLOW
    assert high.request is not None and high.request.status is ApprovalStatus.PENDING
    assert high.request.risk.score == pytest.approx(0.8)


def test_disabled_gate_allows_but_records_why(registry):
    gate = ApprovalGate(enabled=False)
    outcome = gate.evaluate(
        "run_1", 1, registry.get("delete_customer_record"), {"customer_id": "CUST-001"}, True
    )
    assert outcome.decision is RiskDecision.ALLOW and outcome.request is None
    assert outcome.risk.reasons[-1] == "approval gate disabled"


def test_decision_rules(registry):
    gate = ApprovalGate()
    spec = registry.get("delete_customer_record")
    request = gate.evaluate("run_1", 1, spec, {"customer_id": "CUST-001"}, False).request
    with pytest.raises(ApprovalError, match="reviewer"):
        gate.decide(request, True, reviewer=" ")
    with pytest.raises(ApprovalError, match="reason"):
        gate.decide(request, False, reviewer="alice")
    gate.decide(request, True, reviewer="alice")
    assert request.status is ApprovalStatus.APPROVED and request.decided_at is not None
    with pytest.raises(ApprovalError, match="already approved"):
        gate.decide(request, False, reviewer="bob", reason="late")
