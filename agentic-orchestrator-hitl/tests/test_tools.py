import pytest
from pydantic import BaseModel

from agentic_orchestrator.toolbox import safe_eval
from agentic_orchestrator.tools import SideEffect, ToolError, ToolRegistry, ToolSpec


def test_json_schema_is_function_calling_shaped(registry):
    schema = registry.get("issue_refund").json_schema()
    assert schema["name"] == "issue_refund"
    assert set(schema["parameters"]["required"]) == {"order_id", "amount"}
    assert schema["parameters"]["properties"]["amount"]["exclusiveMinimum"] == 0


def test_schemas_are_sorted_and_cover_all_tools(registry):
    names = [s["name"] for s in registry.schemas()]
    assert names == sorted(names)
    assert len(names) == 6


def test_duplicate_registration_is_rejected():
    class Args(BaseModel):
        x: int

    registry = ToolRegistry()
    spec = ToolSpec("t", "d", Args, lambda a: a.x)
    registry.register(spec)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(spec)


def test_invalid_arguments_become_error_result_with_field_path(registry, backend):
    result = registry.execute("issue_refund", {"order_id": "1002", "amount": -5})
    assert not result.ok
    assert "order_id" in result.error and "amount" in result.error
    assert backend.refunds == []


def test_unknown_tool_lists_available_tools(registry):
    with pytest.raises(ToolError, match="calculator"):
        registry.validate("format_disk", {})


def test_tool_error_is_captured_not_raised(registry):
    result = registry.execute("lookup_order", {"order_id": "ORD-9999"})
    assert not result.ok
    assert result.as_text() == "ERROR: order ORD-9999 not found"


def test_refund_cannot_exceed_order_total(registry, backend):
    assert registry.execute("issue_refund", {"order_id": "ORD-1001", "amount": 30}).ok
    second = registry.execute("issue_refund", {"order_id": "ORD-1001", "amount": 30})
    assert not second.ok and "exceeds refundable 10.00" in second.error
    assert len(backend.refunds) == 1


def test_side_effect_metadata(registry):
    assert registry.get("kb_search").side_effect is SideEffect.NONE
    assert registry.get("delete_customer_record").side_effect is SideEffect.DESTRUCTIVE
    assert registry.get("delete_customer_record").reversible is False


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("17 * 23 + 4", 395.0), ("(1200 - 300) / 4", 225.0), ("-2 ** 3", -8.0), ("7 % 4", 3.0)],
)
def test_safe_eval_arithmetic(expression, expected):
    assert safe_eval(expression) == expected


@pytest.mark.parametrize("expression", ["__import__('os')", "a + 1", "1 / 0", "2 ** 999", "1 +"])
def test_safe_eval_rejects_unsafe_or_invalid_input(expression):
    with pytest.raises(ToolError):
        safe_eval(expression)
