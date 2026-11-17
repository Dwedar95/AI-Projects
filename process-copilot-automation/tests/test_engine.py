import json

import pytest

from process_copilot.engine import WorkflowEngine
from process_copilot.policy import ApprovalRequest, PolicyEngine, StaticApprover
from process_copilot.process import ApprovalRule, ProcessDefinition
from process_copilot.tools import ToolError

from .conftest import find_invoice, find_order


def eligible(o):
    return o["status"] == "delivered" and o["days_since_delivery"] <= 30


def refund_msg(order):
    return {"customer_id": order["customer_id"], "message": f"I would like a refund for order {order['order_id']}."}


def steps(trace):
    return [s.step_id for s in trace.steps]


def test_small_refund_runs_straight_through(make_engine, systems_registry):
    _, systems = systems_registry
    order = find_order(systems, lambda o: eligible(o) and o["amount"] <= 100)
    result = make_engine("customer_refund").run(refund_msg(order))
    t = result.trace
    assert t.status == "completed" and t.automated and t.interventions == 0
    assert steps(t) == ["classify_intent", "route_intent", "extract_details", "lookup_order", "check_policy",
                        "check_eligibility", "issue_refund", "draft_confirmation", "close_ticket"]
    assert systems.refunds == [{"refund_id": "RF-0001", "order_id": order["order_id"], "amount": order["amount"],
                                "status": "issued"}]
    assert systems.tickets[0]["status"] == "resolved" and "RF-0001" in systems.tickets[0]["body"]
    assert result.context["policy"]["sources"][0].startswith("refund_policy")
    assert t.human_minutes == 0 and t.manual_minutes == pytest.approx(10.0)


def test_refund_above_threshold_needs_and_records_approval(make_engine, systems_registry):
    _, systems = systems_registry
    order = find_order(systems, lambda o: eligible(o) and 100 < o["amount"] <= 500)
    t = make_engine("customer_refund").run(refund_msg(order)).trace
    issue = next(s for s in t.steps if s.step_id == "issue_refund")
    assert issue.approval.startswith("approved by simulated reviewer (Team lead")
    assert issue.human_minutes == 2.0
    assert t.status == "completed" and not t.automated and t.interventions == 1
    assert len(systems.refunds) == 1


def test_denied_approval_hands_off_without_side_effect(make_engine, systems_registry):
    _, systems = systems_registry
    order = find_order(systems, lambda o: eligible(o) and o["amount"] > 500)
    t = make_engine("customer_refund").run(refund_msg(order)).trace
    assert t.status == "handoff" and "approval denied" in t.handoff_reason
    assert t.steps[-1].status == "denied"
    assert systems.refunds == [] and systems.tickets == []
    assert t.human_minutes == pytest.approx(2.0 + 9.0)


def test_engine_fails_closed_without_approver(make_engine, systems_registry):
    _, systems = systems_registry
    order = find_order(systems, lambda o: eligible(o) and 100 < o["amount"] <= 500)
    t = make_engine("customer_refund", approver=None).run(refund_msg(order)).trace
    assert t.status == "handoff" and "no approver" in t.handoff_reason
    assert systems.refunds == []


def test_ineligible_order_takes_rejection_branch(make_engine, systems_registry):
    _, systems = systems_registry
    order = find_order(systems, lambda o: o["days_since_delivery"] > 30)
    result = make_engine("customer_refund").run(refund_msg(order))
    assert "draft_rejection" in steps(result.trace) and "issue_refund" not in steps(result.trace)
    assert "30 days" in result.context["reply"]["text"]
    assert systems.refunds == []


def test_cascade_to_strong_model_on_low_confidence(make_engine, systems_registry):
    _, systems = systems_registry
    order = find_order(systems, lambda o: eligible(o) and o["amount"] <= 100)
    msg = {"customer_id": "C-1001", "message": f"Order {order['order_id']} was faulty and I want it refunded."}
    t = make_engine("customer_refund").run(msg).trace
    first = t.steps[0]
    assert first.escalated and first.model == "prop-large" and "escalated" in first.routing_reason
    assert first.confidence == 1.0 and t.status == "completed"
    extract = next(s for s in t.steps if s.step_id == "extract_details")
    assert extract.model == "prop-small" and not extract.escalated


def test_low_confidence_and_missing_data_hand_off(make_engine):
    engine = make_engine("customer_refund")
    unclear = engine.run({"customer_id": "C-1001", "message": "Hello, I have a question about my account."}).trace
    assert unclear.status == "handoff" and unclear.handoff_reason.startswith("classify_intent: confidence 0.00")
    no_id = engine.run({"customer_id": "C-1001", "message": "I want a refund for my last order."}).trace
    assert no_id.status == "handoff" and no_id.handoff_reason.startswith("extract_details")
    assert no_id.steps[-1].escalated  # the strong model was tried before giving up


def test_non_refund_intent_goes_to_human_step(make_engine):
    t = make_engine("customer_refund").run(
        {"customer_id": "C-1001", "message": "Where is my order ORD-1001? The tracking is stuck."}).trace
    assert steps(t) == ["classify_intent", "route_intent", "other_request"]
    assert t.status == "completed" and not t.automated and t.human_minutes == 6.0


def test_invoice_within_tolerance_is_paid(make_engine, systems_registry):
    _, systems = systems_registry
    inv = find_invoice(systems, lambda i: i["note"].startswith("Invoice matches") and i["amount"] <= 5000)
    t = make_engine("invoice_exception").run({"invoice_id": inv["invoice_id"]}).trace
    assert steps(t) == ["get_invoice", "check_po_reference", "get_po", "match_amounts", "approve_payment"]
    assert t.automated and t.cost_usd == 0.0  # no model call needed on the clean path
    assert systems.payments[0]["invoice_id"] == inv["invoice_id"]


def test_large_payment_needs_controller_approval(make_engine, systems_registry):
    _, systems = systems_registry
    inv = find_invoice(systems, lambda i: i["note"].startswith("Invoice matches") and 5000 < i["amount"] <= 15000)
    t = make_engine("invoice_exception").run({"invoice_id": inv["invoice_id"]}).trace
    assert "Finance controller" in t.steps[-1].approval and t.steps[-1].human_minutes == 3.0


def test_invoice_exception_is_prepared_for_the_clerk_on_self_hosted_models(make_engine, systems_registry):
    _, systems = systems_registry
    inv = find_invoice(systems, lambda i: i["note"].startswith("Unit price"))
    result = make_engine("invoice_exception").run({"invoice_id": inv["invoice_id"]})
    t = result.trace
    assert result.context["exception"]["label"] == "price_variance"
    assert result.context["guidance"]["sources"][0].startswith("invoice_policy")
    assert steps(t)[-2:] == ["open_procurement_ticket", "resolve_exception"]
    assert {s.hosting for s in t.steps if s.model} == {"open_source"}
    assert next(s for s in t.steps if s.step_id == "summarise_case").model == "oss-large"
    assert systems.tickets[0]["queue"] == "procurement" and systems.payments == []
    assert t.human_minutes == 8.0 and t.manual_minutes == pytest.approx(21.5)


def test_invoice_without_po_is_classified(make_engine, systems_registry):
    _, systems = systems_registry
    inv = find_invoice(systems, lambda i: i["po_id"] == "")
    result = make_engine("invoice_exception").run({"invoice_id": inv["invoice_id"]})
    assert "get_po" not in steps(result.trace)
    assert result.context["exception"]["label"] == "missing_purchase_order"


def test_hr_sensitive_topic_goes_to_hr_partner_without_generated_answer(make_engine, systems_registry):
    _, systems = systems_registry
    result = make_engine("hr_onboarding").run({"employee_id": "E-3001", "question": "I need to report harassment."})
    assert steps(result.trace)[-1] == "hr_partner" and "answer" not in result.context
    assert systems.tickets == []


def test_hr_question_answered_from_handbook_and_it_task_created(make_engine, systems_registry):
    _, systems = systems_registry
    engine = make_engine("hr_onboarding")
    vac = engine.run({"employee_id": "E-3001", "question": "How many vacation days do I have in my first year?"})
    assert "28 vacation days" in vac.context["answer"]["answer"] and vac.trace.automated
    it = engine.run({"employee_id": "E-3002", "question": "When do I get my laptop and VPN access?"})
    assert "create_it_task" in steps(it.trace)
    assert systems.onboarding_tasks[0]["employee_id"] == "E-3002"
    assert all(s.hosting == "open_source" for s in it.trace.steps if s.model)


def test_budget_exhaustion_hands_off(make_engine):
    t = make_engine("hr_onboarding", cost_budget_usd=1e-9).run(
        {"employee_id": "E-3001", "question": "How many vacation days do I have?"}).trace
    assert t.status == "handoff" and "BudgetExceededError" in t.handoff_reason


def test_trace_totals_and_serialisation(make_engine, systems_registry):
    _, systems = systems_registry
    order = find_order(systems, lambda o: eligible(o) and o["amount"] <= 100)
    t = make_engine("customer_refund").run(refund_msg(order), case_id="c-9").trace
    assert t.cost_usd == pytest.approx(sum(s.cost_usd for s in t.steps)) and t.cost_usd > 0
    assert t.cost_usd <= 0.02
    d = json.loads(json.dumps(t.to_dict()))
    assert d["case_id"] == "c-9" and d["automated"] is True and len(d["steps"]) == 9
    assert "classify_intent" in t.table()
    assert all(s.wall_ms >= 0 for s in t.steps)


def _custom(steps_data, **kw):
    return ProcessDefinition.model_validate({"id": "x", "name": "X", "steps": steps_data, **kw})


def test_tool_error_missing_input_and_loop_guard(systems_registry):
    registry, systems = systems_registry
    p = _custom([{"id": "refund", "type": "action", "tool": "finance.issue_refund",
                  "args": {"order_id": "{{ input.order_id }}", "amount": -5}}],
                allowed_tools=["finance.issue_refund"], inputs=["order_id"])
    engine = WorkflowEngine(p, registry)
    t = engine.run({"order_id": "ORD-1000"}).trace
    assert t.status == "handoff" and "ToolError" in t.handoff_reason and systems.refunds == []
    with pytest.raises(ValueError, match="missing inputs"):
        engine.run({})
    loop = _custom([{"id": "a", "type": "decision", "branches": [{"when": "1 == 1", "goto": "a"}], "default": "end"}])
    t = WorkflowEngine(loop, registry, max_steps=5).run({}).trace
    assert t.status == "handoff" and "step limit" in t.handoff_reason
    with pytest.raises(ToolError):
        WorkflowEngine(_custom([{"id": "a", "type": "human"}], allowed_tools=["nope.tool"]), registry)


def test_policy_engine_first_match_and_registry_errors(systems_registry):
    registry, _ = systems_registry
    rules = [ApprovalRule(tool="finance.issue_refund", when="args.amount > 100", approver_role="Lead"),
             ApprovalRule(tool="*", when="input.vip", approver_role="Manager")]
    policy = PolicyEngine(rules)
    ctx = {"input": {"vip": False}}
    assert policy.requirement("finance.issue_refund", {"amount": 150}, ctx).approver_role == "Lead"
    assert policy.requirement("finance.issue_refund", {"amount": 50}, ctx) is None
    assert policy.requirement("crm.log_interaction", {}, {"input": {"vip": True}}).approver_role == "Manager"
    with pytest.raises(ToolError, match="unknown tool"):
        registry.call("nope", {})
    with pytest.raises(ToolError, match="bad arguments"):
        registry.call("crm.get_order", {"wrong": 1})
    decision = StaticApprover(False).decide(ApprovalRequest("c", "s", "t", {}, "Lead", "", 1.0))
    assert decision.approved is False
