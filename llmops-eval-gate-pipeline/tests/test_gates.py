import pytest
from pydantic import ValidationError

from llmops_gate.gates import GateConfig, GateRule, evaluate_gates, load_gate_config

from .conftest import BENCHMARKS, GATES


def _config(**rule) -> GateConfig:
    return GateConfig(gates=[GateRule(**rule)])


def test_absolute_limits():
    config = GateConfig(
        gates=[
            GateRule(metric="accuracy", min=0.8),
            GateRule(metric="latency_p95_ms", direction="lower_is_better", max=50),
        ]
    )
    assert evaluate_gates({"accuracy": 0.8, "latency_p95_ms": 50.0}, None, config).passed
    decision = evaluate_gates({"accuracy": 0.79, "latency_p95_ms": 51.0}, None, config)
    assert not decision.passed and len(decision.failures()) == 2
    assert decision.failures()[0] == "accuracy 0.79 >= min 0.8"


def test_absolute_regression_for_higher_is_better():
    config = _config(metric="accuracy", max_regression=0.02)
    assert evaluate_gates({"accuracy": 0.88}, {"accuracy": 0.90}, config).passed
    assert evaluate_gates({"accuracy": 0.95}, {"accuracy": 0.90}, config).passed
    decision = evaluate_gates({"accuracy": 0.87}, {"accuracy": 0.90}, config, "model:x:v1")
    assert not decision.passed and decision.baseline_ref == "model:x:v1"
    assert decision.checks[0].limit == pytest.approx(0.88)


def test_relative_regression_for_lower_is_better():
    config = _config(metric="latency_p95_ms", direction="lower_is_better", max_regression_pct=20)
    assert evaluate_gates({"latency_p95_ms": 119.0}, {"latency_p95_ms": 100.0}, config).passed
    assert evaluate_gates({"latency_p95_ms": 60.0}, {"latency_p95_ms": 100.0}, config).passed
    decision = evaluate_gates({"latency_p95_ms": 121.0}, {"latency_p95_ms": 100.0}, config)
    assert not decision.passed and "21.0%" in decision.failures()[0]


def test_regression_rules_are_skipped_without_baseline():
    config = _config(metric="accuracy", min=0.5, max_regression=0.0)
    decision = evaluate_gates({"accuracy": 0.6}, None, config)
    assert decision.passed and [c.rule for c in decision.checks] == ["min"]
    # A baseline that lacks the metric is treated the same way.
    assert evaluate_gates({"accuracy": 0.6}, {"other": 1.0}, config).passed


def test_missing_candidate_metric_fails_the_gate():
    decision = evaluate_gates({}, None, _config(metric="accuracy", min=0.5))
    assert not decision.passed and "missing" in decision.failures()[0]


def test_rule_without_any_check_is_invalid():
    with pytest.raises(ValidationError, match="defines no check"):
        GateRule(metric="accuracy")
    with pytest.raises(ValidationError):
        GateConfig(gates=[])


def test_committed_gate_configs_load():
    assert [g.metric for g in load_gate_config(GATES).gates][:2] == ["accuracy", "macro_f1"]
    prompt = load_gate_config(BENCHMARKS / "prompt_gates.yaml")
    assert any(g.max_regression_pct == 15 for g in prompt.gates)
