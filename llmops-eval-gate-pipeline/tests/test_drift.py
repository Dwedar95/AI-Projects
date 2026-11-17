import math

import numpy as np
import pytest

from llmops_gate.datasets import generate_tickets
from llmops_gate.drift import (
    AlertRule,
    categorical_psi,
    embedding_drift,
    evaluate_alerts,
    ks_test,
    load_alert_rules,
    monitor,
    psi,
)

from .conftest import BENCHMARKS


def test_psi_is_near_zero_for_the_same_distribution_and_large_for_a_shift():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 5000)
    assert psi(reference, rng.normal(0, 1, 5000)) < 0.02
    assert psi(reference, rng.normal(1.0, 1, 5000)) > 0.5
    assert psi(reference, reference) == pytest.approx(0.0, abs=1e-12)


def test_psi_matches_a_hand_computed_two_bin_case():
    reference = np.array([0.0] * 50 + [1.0] * 50)
    current = np.array([0.0] * 75 + [1.0] * 25)
    expected = (0.75 - 0.5) * math.log(0.75 / 0.5) + (0.25 - 0.5) * math.log(0.25 / 0.5)
    assert psi(reference, current, bins=2) == pytest.approx(expected)


def test_psi_input_validation():
    with pytest.raises(ValueError):
        psi(np.arange(5), np.arange(100), bins=10)
    with pytest.raises(ValueError):
        psi(np.arange(100), np.array([]))


def test_categorical_psi_hand_computed_and_new_category():
    value = categorical_psi(["a", "a", "b", "b"], ["a", "a", "a", "b"])
    assert value == pytest.approx(0.25 * math.log(1.5) - 0.25 * math.log(0.5))
    assert categorical_psi(["a"] * 10, ["a"] * 5 + ["z"] * 5) > 1.0


def test_ks_test_separates_shifted_samples():
    rng = np.random.default_rng(1)
    same = ks_test(rng.normal(size=800), rng.normal(size=800))
    shifted = ks_test(rng.normal(size=800), rng.normal(0.5, 1, size=800))
    assert same[1] > 0.01 and shifted[1] < 1e-6 and shifted[0] > same[0]


def test_embedding_drift_detects_a_mean_shift():
    rng = np.random.default_rng(2)
    reference = rng.normal(size=(400, 32))
    stable = embedding_drift(reference, rng.normal(size=(400, 32)))
    moved = embedding_drift(reference, rng.normal(size=(400, 32)) + 0.8)
    assert stable["drifted_component_share"] <= 0.125
    assert moved["drifted_component_share"] >= 0.5
    assert moved["centroid_distance_psi"] > stable["centroid_distance_psi"]
    with pytest.raises(ValueError):
        embedding_drift(reference, rng.normal(size=(10, 8)))


def test_alert_rules_cover_both_directions_and_severities():
    rules = [
        AlertRule(metric="score_psi", warn=0.1, critical=0.25),
        AlertRule(metric="score_ks_pvalue", warn=0.01, critical=0.0001, direction="below"),
        AlertRule(metric="not_reported", warn=0, critical=0),
    ]
    alerts = evaluate_alerts({"score_psi": 0.2, "score_ks_pvalue": 1e-9}, rules)
    assert [(a.metric, a.severity) for a in alerts] == [
        ("score_psi", "warning"),
        ("score_ks_pvalue", "critical"),
    ]
    assert evaluate_alerts({"score_psi": 0.05, "score_ks_pvalue": 0.5}, rules) == []


def _logs(tickets, confidence_shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    scores = np.clip(rng.beta(8, 2, len(tickets)) - confidence_shift, 0, 1)
    return [
        {"text": t["text"], "label": t["label"], "score": float(s), "latency_ms": 1.0}
        for t, s in zip(tickets, scores, strict=True)
    ]


def test_monitor_end_to_end_ok_and_critical():
    rules = load_alert_rules(BENCHMARKS / "alert_rules.yaml")
    reference = _logs(generate_tickets(400, seed=7), seed=1)
    stable = monitor(reference, _logs(generate_tickets(400, seed=8), seed=2), rules)
    assert stable.status == "ok" and stable.metrics["latency_p95_ratio"] == 1.0
    shifted_tickets = generate_tickets(
        400, seed=9, priors=[0.1, 0.1, 0.6, 0.1, 0.1], new_term_rate=0.5
    )
    shifted = monitor(reference, _logs(shifted_tickets, confidence_shift=0.2, seed=3), rules)
    assert shifted.status == "critical"
    fired = {a.metric for a in shifted.alerts}
    assert {"score_psi", "label_psi", "drifted_component_share"} <= fired


def test_monitor_needs_enough_samples():
    logs = _logs(generate_tickets(10, seed=0))
    with pytest.raises(ValueError, match="at least 30"):
        monitor(logs, logs, [])
