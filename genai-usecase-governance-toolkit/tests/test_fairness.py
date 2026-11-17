import numpy as np
import pytest

from genai_governance.fairness import (
    PredictionData,
    confusion,
    fairness_report,
    load_predictions,
    performance_metrics,
    roc_auc,
)


def make(y_true, y_pred, group, score=None):
    return PredictionData(np.array(y_true), np.array(y_pred), np.array(group),
                          None if score is None else np.array(score))


def test_confusion_and_performance():
    d = make([1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 1, 0, 0, 0, 0], list("aaaabbbb"))
    assert confusion(d.y_true, d.y_pred) == (2, 1, 4, 1)
    p = performance_metrics(d)
    assert p.accuracy == pytest.approx(6 / 8)
    assert p.precision == pytest.approx(2 / 3)
    assert p.recall == pytest.approx(2 / 3)
    assert p.specificity == pytest.approx(4 / 5)
    assert p.f1 == pytest.approx(2 / 3)
    assert p.roc_auc is None


def test_roc_auc_known_values_and_ties():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert roc_auc(y, np.array([0.1, 0.4, 0.35, 0.8])) == pytest.approx(0.75)
    assert roc_auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)


def test_fairness_metrics_hand_computed():
    # group a: selection 3/4, TPR 2/2, FPR 1/2. group b: selection 1/4, TPR 1/2, FPR 0/2.
    d = make([1, 1, 0, 0, 1, 1, 0, 0], [1, 1, 1, 0, 1, 0, 0, 0], list("aaaabbbb"))
    r = fairness_report(d, min_group_size=1)
    assert r.demographic_parity_difference == pytest.approx(0.5)
    assert r.tpr_difference == pytest.approx(0.5)
    assert r.fpr_difference == pytest.approx(0.5)
    assert r.equalised_odds_difference == pytest.approx(0.5)
    assert r.disparate_impact == pytest.approx(1 / 3)
    assert len(r.flags) == 3


def test_fair_predictions_raise_no_flags():
    d = make([1, 0, 1, 0], [1, 0, 1, 0], list("aabb"))
    r = fairness_report(d, min_group_size=1)
    assert r.demographic_parity_difference == 0 and r.disparate_impact == 1.0
    assert r.flags == []


def test_small_group_is_flagged():
    d = make([1, 0, 1, 0], [1, 0, 1, 0], list("aabb"))
    assert any("unreliable" in f for f in fairness_report(d).flags)


def test_three_groups_use_max_minus_min():
    d = make([1] * 6, [1, 1, 1, 0, 0, 0], list("aabbcc"))
    r = fairness_report(d, min_group_size=1)
    assert r.demographic_parity_difference == pytest.approx(1.0)
    assert r.disparate_impact == 0.0


def test_input_validation():
    with pytest.raises(ValueError):
        make([1, 2], [1, 0], ["a", "b"])
    with pytest.raises(ValueError):
        make([1, 0], [1], ["a", "b"])
    with pytest.raises(ValueError):
        fairness_report(make([1, 0], [1, 0], ["a", "a"]))


def test_synthetic_file_shows_injected_bias(predictions_path):
    data = load_predictions(predictions_path)
    assert len(data.y_true) == 2000 and data.y_score is not None
    r = fairness_report(data, bootstrap=200, seed=1)
    by = {g.group: g for g in r.groups}
    assert by["A"].selection_rate > by["B"].selection_rate
    assert r.dpd_ci95[0] <= r.demographic_parity_difference <= r.dpd_ci95[1]
    assert r == fairness_report(data, bootstrap=200, seed=1)
    assert 0.5 < performance_metrics(data).roc_auc < 1.0


def test_load_predictions_reports_missing_column(tmp_path):
    f = tmp_path / "p.csv"
    f.write_text("y_true,y_pred\n1,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="group"):
        load_predictions(f)
