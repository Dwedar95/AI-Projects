from datetime import UTC, datetime

import pytest

from llmops_gate.cost import (
    BudgetAction,
    BudgetGovernor,
    PricingTable,
    TokenBudget,
    UsageTracker,
    load_budgets,
)

SEPT = datetime(2026, 9, 15, tzinfo=UTC).timestamp()
OCT = datetime(2026, 10, 1, tzinfo=UTC).timestamp()


def test_cost_formula_and_unknown_model():
    pricing = PricingTable.example()
    assert pricing.cost("large-llm", 1000, 1000) == pytest.approx(0.02)
    assert pricing.cost("small-llm", 2000, 0) == pytest.approx(0.0006)
    assert pricing.cheapest() == "onnx-classifier"
    with pytest.raises(KeyError):
        pricing.cost("mystery-model", 1, 1)


def test_summary_aggregates_per_team_and_model():
    tracker = UsageTracker(PricingTable.example())
    for latency in (100.0, 200.0, 300.0):
        tracker.record("a", "large-llm", 1000, 100, latency, ts=SEPT)
    tracker.record("b", "small-llm", 500, 50, 50.0, ts=SEPT)
    by_team = {row["team"]: row for row in tracker.summary("team")}
    assert by_team["a"]["requests"] == 3 and by_team["a"]["input_tokens"] == 3000
    assert by_team["a"]["cost_usd"] == pytest.approx(3 * 0.0065)
    assert by_team["a"]["cost_per_1k_requests_usd"] == pytest.approx(6.5)
    assert by_team["a"]["latency_p50_ms"] == 200.0 and by_team["a"]["latency_p95_ms"] == 290.0
    assert [row["model"] for row in tracker.summary("model")] == ["large-llm", "small-llm"]
    with pytest.raises(ValueError):
        tracker.summary("region")


def test_usage_log_persists_and_reloads(tmp_path):
    path = tmp_path / "logs" / "usage.jsonl"
    UsageTracker(PricingTable.example(), path).record("a", "small-llm", 100, 10, 5.0, ts=SEPT)
    reloaded = UsageTracker(PricingTable.example(), path)
    reloaded.record("a", "small-llm", 100, 10, 5.0, ts=SEPT)
    assert len(path.read_text().splitlines()) == 2
    assert reloaded.month_totals("a", "2026-09")[0] == 220


def _governor(fallback="small-llm"):
    tracker = UsageTracker(PricingTable.example())
    budgets = {"a": TokenBudget(monthly_tokens=10_000, monthly_usd=1.0)}
    return tracker, BudgetGovernor(tracker, budgets, fallback_model=fallback)


def test_budget_allow_then_downgrade_then_deny():
    tracker, governor = _governor()
    first = governor.check("a", "large-llm", 1000, now=SEPT)
    assert first.action is BudgetAction.ALLOW and first.utilisation == pytest.approx(0.1)
    tracker.record("a", "large-llm", 7500, 0, 10.0, ts=SEPT)
    soft = governor.check("a", "large-llm", 1000, now=SEPT)
    assert soft.action is BudgetAction.DOWNGRADE and soft.model == "small-llm"
    tracker.record("a", "small-llm", 2000, 0, 10.0, ts=SEPT)
    hard = governor.check("a", "large-llm", 1000, now=SEPT)
    assert hard.action is BudgetAction.DENY and hard.model is None and hard.utilisation > 1.0


def test_usd_limit_binds_independently_of_the_token_limit():
    tracker = UsageTracker(PricingTable.example())
    budgets = {"a": TokenBudget(monthly_tokens=10_000_000, monthly_usd=0.01)}
    governor = BudgetGovernor(tracker, budgets)
    tracker.record("a", "large-llm", 1000, 400, 10.0, ts=SEPT)  # 0.011 USD
    assert governor.check("a", "large-llm", 10, now=SEPT).action is BudgetAction.DENY


def test_budget_resets_with_the_calendar_month():
    tracker, governor = _governor()
    tracker.record("a", "large-llm", 9900, 0, 10.0, ts=SEPT)
    assert governor.check("a", "large-llm", 500, now=SEPT).action is BudgetAction.DENY
    assert governor.check("a", "large-llm", 500, now=OCT).action is BudgetAction.ALLOW


def test_no_downgrade_when_already_on_the_cheap_model_or_no_fallback():
    tracker, governor = _governor()
    tracker.record("a", "small-llm", 8500, 0, 10.0, ts=SEPT)
    assert governor.check("a", "small-llm", 100, now=SEPT).action is BudgetAction.ALLOW
    tracker2, no_fallback = _governor(fallback=None)
    tracker2.record("a", "large-llm", 8500, 0, 10.0, ts=SEPT)
    assert no_fallback.check("a", "large-llm", 100, now=SEPT).action is BudgetAction.ALLOW


def test_team_without_budget_is_denied():
    _, governor = _governor()
    decision = governor.check("unknown-team", "small-llm", 10, now=SEPT)
    assert decision.action is BudgetAction.DENY and decision.reason == "no budget configured"


def test_load_budgets_validates(tmp_path):
    path = tmp_path / "budgets.json"
    path.write_text('{"a": {"monthly_tokens": 100, "monthly_usd": 1.5}}')
    assert load_budgets(path)["a"].soft_ratio == 0.8
    path.write_text('{"a": {"monthly_tokens": 0, "monthly_usd": 1.5}}')
    with pytest.raises(ValueError):
        load_budgets(path)
