import pytest
from pydantic import ValidationError

from process_copilot.expressions import ExpressionError, evaluate, render
from process_copilot.process import ProcessDefinition, load_process

from .conftest import PROCESSES

CTX = {"order": {"amount": 120.5, "status": "delivered", "tags": ["a", "b"]}, "input": {"message": "hi"}}


def test_render_keeps_type_for_single_placeholder_and_recurses():
    assert render("{{ order.amount }}", CTX) == 120.5
    assert render("Amount: {{ order.amount }} USD", CTX) == "Amount: 120.5 USD"
    assert render({"a": ["{{ input.message }}", 3]}, CTX) == {"a": ["hi", 3]}


def test_render_missing_path_raises():
    with pytest.raises(ExpressionError, match="missing 'nope'"):
        render("{{ order.nope }}", CTX)


def test_evaluate_comparisons_logic_and_arithmetic():
    assert evaluate("order.amount > 100 and order.status == 'delivered'", CTX) is True
    assert evaluate("not (order.amount > 100) or 'a' in order.tags", CTX) is True
    assert evaluate("abs(order.amount - 130) / 130 <= 0.1", CTX) is True
    assert evaluate("1 < 2 < 3", CTX) is True
    assert evaluate("order.status in ['open', 'closed']", CTX) is False


@pytest.mark.parametrize("expr", [
    "__import__('os').system('true')",
    "order.__class__",
    "open('x')",
    "[x for x in order.tags]",
    "lambda: 1",
    "order.amount >",
    "unknown_name == 1",
    "1 / 0 > 1",
])
def test_evaluate_rejects_unsafe_or_invalid_expressions(expr):
    with pytest.raises(ExpressionError):
        evaluate(expr, CTX)


@pytest.mark.parametrize("name", ["customer_refund", "invoice_exception", "hr_onboarding"])
def test_example_processes_load(name):
    p = load_process(PROCESSES / f"{name}.yaml")
    assert p.id == name
    assert p.knowledge_base.endswith("knowledge")
    assert {s.type for s in p.steps} >= {"action", "decision", "human", "prompt", "rag"}
    assert all(s.role for s in p.steps)


def test_successor_follows_next_then_order():
    p = load_process(PROCESSES / "customer_refund.yaml")
    assert p.successor("classify_intent") == "route_intent"
    assert p.successor("draft_confirmation") == "close_ticket"
    assert p.successor("other_request") == "end"


def _minimal(**overrides):
    step = {"id": "a", "type": "human"}
    data = {"id": "p", "name": "P", "steps": [step]}
    data.update(overrides)
    return data


def test_process_validation_errors():
    with pytest.raises(ValidationError, match="unknown step"):
        ProcessDefinition.model_validate(_minimal(steps=[{"id": "a", "type": "human", "next": "missing"}]))
    with pytest.raises(ValidationError, match="unique"):
        ProcessDefinition.model_validate(_minimal(steps=[{"id": "a", "type": "human"}, {"id": "a", "type": "human"}]))
    with pytest.raises(ValidationError, match="allowed_tools"):
        ProcessDefinition.model_validate(_minimal(steps=[{"id": "a", "type": "action", "tool": "crm.get_order"}]))
    with pytest.raises(ValidationError, match="two labels"):
        ProcessDefinition.model_validate(_minimal(steps=[{"id": "a", "type": "prompt", "task": "classify", "prompt": "x"}]))
    with pytest.raises(ValidationError, match="knowledge_base"):
        ProcessDefinition.model_validate(_minimal(steps=[{"id": "a", "type": "rag", "query": "q"}]))
    with pytest.raises(ValidationError, match="reserved"):
        ProcessDefinition.model_validate(_minimal(steps=[{"id": "end", "type": "human"}]))
