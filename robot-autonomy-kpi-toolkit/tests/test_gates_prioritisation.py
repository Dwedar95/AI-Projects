import math

import pytest
import yaml
from pydantic import ValidationError

from autonomy_kpi.gates import (
    GateConfig,
    evaluate_gates,
    fisher_worse,
    load_gate_config,
    mann_whitney_worse,
    overall_verdict,
    poisson_rate_worse,
    relative_regression,
)
from autonomy_kpi.kpis import KPI_NAMES
from autonomy_kpi.prioritisation import BacklogItem, kpi_coverage, load_backlog, rank_backlog

from .conftest import EXAMPLES, make_summary


def _fleet(n, **overrides):
    return [make_summary(f"m{i}", **overrides) for i in range(n)]


def test_relative_regression_sign_convention():
    assert relative_regression(0.9, 0.81, higher_is_better=True) == pytest.approx(0.1)
    assert relative_regression(0.9, 0.99, higher_is_better=True) == pytest.approx(-0.1)
    assert relative_regression(2.0, 3.0, higher_is_better=False) == pytest.approx(0.5)
    assert relative_regression(math.inf, 5.0, higher_is_better=True) == math.inf
    assert relative_regression(5.0, math.inf, higher_is_better=True) == -math.inf
    assert relative_regression(0.0, 0.0, higher_is_better=False) == 0.0
    assert relative_regression(0.0, 1.0, higher_is_better=False) == math.inf


def test_statistical_tests_detect_direction():
    base = _fleet(40)
    worse = _fleet(20) + _fleet(20, outcome="failed")
    assert fisher_worse(base, worse) < 1e-4
    assert fisher_worse(worse, base) > 0.99
    assert poisson_rate_worse(10, 10.0, 30, 10.0) < 0.01
    assert poisson_rate_worse(30, 10.0, 10, 10.0) > 0.95
    # Same rate with different exposure is not a regression.
    assert poisson_rate_worse(10, 10.0, 20, 20.0) > 0.3
    assert poisson_rate_worse(0, 10.0, 0, 10.0) == 1.0
    low, high = [0.1 * i for i in range(20)], [5 + 0.1 * i for i in range(20)]
    assert mann_whitney_worse(high, low, higher_is_better=True) < 1e-4
    assert mann_whitney_worse(low, high, higher_is_better=True) > 0.99
    assert mann_whitney_worse(low, high, higher_is_better=False) < 1e-4
    assert mann_whitney_worse([1.0, 1.0], [1.0, 1.0], True) == 1.0


def _config(**gates):
    return GateConfig(alpha=0.05, gates=gates)


def test_identical_builds_pass_all_example_gates():
    config = load_gate_config(EXAMPLES / "gates.yaml")
    fleet = [make_summary(f"m{i}", interventions=i % 3) for i in range(30)]
    results = evaluate_gates(fleet, fleet, config)
    assert [r.kpi for r in results] == list(config.gates)
    assert all(r.verdict == "PASS" and r.relative_regression == 0.0 for r in results)
    assert overall_verdict(results) == "PASS"


def test_significant_regression_fails():
    base = [make_summary(f"b{i}", interventions=i % 2) for i in range(60)]
    cand = [make_summary(f"c{i}", interventions=2 + i % 2) for i in range(60)]
    (r,) = evaluate_gates(
        base, cand, _config(interventions_per_hour={"max_relative_regression": 0.15})
    )
    assert r.verdict == "FAIL" and r.test == "poisson_rate"
    assert r.relative_regression == pytest.approx(4.0) and r.p_value < 0.001


def test_underpowered_regression_warns_instead_of_failing():
    base = _fleet(5) + _fleet(1, outcome="failed")
    cand = _fleet(4) + _fleet(2, outcome="failed")
    (r,) = evaluate_gates(
        base, cand, _config(mission_success_rate={"max_relative_regression": 0.05})
    )
    assert r.relative_regression == pytest.approx(0.2)
    assert r.p_value > 0.05 and r.verdict == "WARN"
    assert overall_verdict([r]) == "WARN"


def test_absolute_limits_fail_regardless_of_baseline():
    fleet = _fleet(10, min_separation_m=0.2)
    (r,) = evaluate_gates(
        fleet, fleet, _config(min_separation_m={"max_relative_regression": 0.5, "min_value": 0.25})
    )
    assert r.verdict == "FAIL" and "absolute minimum" in r.reason
    busy = _fleet(10, withheld_by_gate=300)
    (r,) = evaluate_gates(busy, busy, _config(gate_withhold_rate={"max_value": 0.10}))
    assert r.verdict == "FAIL" and "absolute maximum" in r.reason


def test_improvement_passes_and_config_validation():
    base = [make_summary(f"b{i}", interventions=3) for i in range(30)]
    cand = [make_summary(f"c{i}", interventions=1) for i in range(30)]
    results = evaluate_gates(
        base, cand, _config(interventions_per_hour={}, mean_distance_between_interventions_km={})
    )
    assert all(r.verdict == "PASS" and r.relative_regression < 0 for r in results)
    with pytest.raises(ValidationError, match="unknown KPIs"):
        _config(robot_happiness={})
    with pytest.raises(ValueError):
        evaluate_gates([], cand, _config(interventions_per_hour={}))


def _item(item_id="X-1", **overrides):
    fields = {
        "id": item_id,
        "title": "t",
        "kpis": ["autonomy_uptime"],
        "reach": 100,
        "impact": 2,
        "confidence": 0.5,
        "effort": 4,
        "business_value": 8,
        "time_criticality": 5,
        "risk_reduction": 3,
        "job_size": 4,
    }
    fields.update(overrides)
    return BacklogItem(**fields)


def test_rice_and_wsjf_formulas():
    item = _item()
    assert item.rice() == pytest.approx(100 * 2 * 0.5 / 4)
    assert item.wsjf() == pytest.approx((8 + 5 + 3) / 4)
    with pytest.raises(ValidationError, match="unknown KPIs"):
        _item(kpis=["velocity"])
    with pytest.raises(ValidationError):
        _item(confidence=1.5)


def test_ranking_order_boost_and_ties():
    items = [
        _item("A", reach=100),
        _item("B", reach=300, kpis=["gate_withhold_rate"]),
        _item("C", reach=100),
        _item("D", reach=90, kpis=["interventions_per_hour"]),
    ]
    plain = rank_backlog(items, "rice")
    assert [r.item.id for r in plain] == ["B", "A", "C", "D"]  # A before C: backlog order on tie
    assert [r.rank for r in plain] == [1, 2, 3, 4]
    boosted = rank_backlog(items, "rice", ["interventions_per_hour"], gate_boost=2.0)
    assert [r.item.id for r in boosted] == ["B", "D", "A", "C"]
    d = boosted[1]
    assert d.addresses == ("interventions_per_hour",) and d.score == 2 * d.base_score
    assert rank_backlog(items, "wsjf")[0].score == pytest.approx(4.0)
    with pytest.raises(ValueError):
        rank_backlog(items, "moscow")
    with pytest.raises(ValueError):
        rank_backlog(items, "rice", gate_boost=0.5)


def test_example_backlog_loads_and_links_to_kpis(tmp_path):
    items = load_backlog(EXAMPLES / "backlog.yaml")
    assert len(items) == 7
    coverage = kpi_coverage(items)
    assert set(coverage) == set(KPI_NAMES)
    assert "AUT-102" in coverage["localisation_error_mean_m"]
    assert all(coverage[k] for k in KPI_NAMES), "every KPI has at least one backlog item"
    dup = tmp_path / "dup.yaml"
    dup.write_text(yaml.safe_dump({"backlog": [_item().model_dump(), _item().model_dump()]}))
    with pytest.raises(ValueError, match="duplicate"):
        load_backlog(dup)
    empty = tmp_path / "empty.yaml"
    empty.write_text("items: []\n")
    with pytest.raises(ValueError, match="top-level"):
        load_backlog(empty)
