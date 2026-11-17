import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from anomalybags.metrics import (
    anomaly_segments,
    best_f1,
    evaluate,
    point_adjust,
    pr_auc,
    roc_auc,
)


def test_auc_metrics_match_sklearn_with_ties(rng):
    y = (rng.random(500) < 0.1).astype(int)
    s = np.round(rng.normal(size=500) + 1.5 * y, 1)  # rounding creates ties
    assert roc_auc(y, s) == pytest.approx(roc_auc_score(y, s))
    assert pr_auc(y, s) == pytest.approx(average_precision_score(y, s))


def test_perfect_and_inverted_scores():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc(y, [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert best_f1(y, [0.1, 0.2, 0.8, 0.9]) == (1.0, 0.8)


def test_best_f1_matches_brute_force(rng):
    y = (rng.random(200) < 0.2).astype(int)
    s = rng.normal(size=200) + y
    brute = 0.0
    for t in np.unique(s):
        pred = s >= t
        tp = np.sum(pred & (y == 1))
        if tp:
            p, r = tp / pred.sum(), tp / y.sum()
            brute = max(brute, 2 * p * r / (p + r))
    assert best_f1(y, s)[0] == pytest.approx(brute)


def test_segments_and_point_adjust():
    y = np.array([0, 1, 1, 0, 0, 1, 0])
    s = np.array([0.0, 0.1, 0.9, 0.0, 0.0, 0.3, 0.0])
    assert anomaly_segments(y) == [(1, 3), (5, 6)]
    assert point_adjust(y, s).tolist() == [0.0, 0.9, 0.9, 0.0, 0.0, 0.3, 0.0]


def test_point_adjust_never_lowers_f1(rng):
    y = np.zeros(300, dtype=int)
    y[50:80] = 1
    y[200:230] = 1
    s = rng.random(300)
    result = evaluate(y, s, with_point_adjust=True)
    assert result["pa_best_f1"] >= result["best_f1"]
    assert result["pa_best_f1"] > 0.5  # random scores look good: why it must be flagged
    assert "pa_best_f1" not in evaluate(y, s)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        roc_auc([0, 0, 0], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        pr_auc([0, 1], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        roc_auc([0, 1], [0.1, np.nan])
