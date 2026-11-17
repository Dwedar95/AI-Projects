"""Evaluation metrics implemented in numpy.

ROC-AUC, PR-AUC (average precision) and best F1 are threshold-free summaries of a
score vector. ``point_adjust`` implements the widely used but optimistic protocol in
which one hit inside a true anomaly segment counts the whole segment as detected
(Xu et al., 2018). Kim et al. (2022) showed that it can make random scores look
good, so every result produced with it is reported under a ``pa_`` prefix and must
not be compared with unadjusted numbers.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def _check(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels).astype(int).ravel()
    s = np.asarray(scores, dtype=float).ravel()
    if y.size != s.size:
        raise ValueError("labels and scores must have the same length")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores contain non-finite values")
    if y.min() == y.max():
        raise ValueError("labels must contain both classes")
    return y, s


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic (ties get mid-ranks)."""
    y, s = _check(labels, scores)
    ranks = rankdata(s)
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    u = ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def precision_recall_curve(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precision and recall at every distinct threshold, in order of decreasing threshold."""
    y, s = _check(labels, scores)
    order = np.argsort(-s, kind="stable")
    y_sorted, s_sorted = y[order], s[order]
    last = np.r_[np.flatnonzero(np.diff(s_sorted) != 0), y.size - 1]
    tp = np.cumsum(y_sorted)[last]
    predicted = last + 1
    precision = tp / predicted
    recall = tp / y.sum()
    return precision, recall, s_sorted[last]


def pr_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Average precision: sum of precision weighted by the recall increments."""
    precision, recall, _ = precision_recall_curve(labels, scores)
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def best_f1(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Best F1 over all thresholds and the threshold that achieves it.

    This is an oracle threshold chosen with the labels, so it is an upper bound on
    what a deployed threshold would reach.
    """
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(np.argmax(f1))
    return float(f1[best]), float(thresholds[best])


def anomaly_segments(labels: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs of positive labels as ``(start, end_exclusive)`` pairs."""
    y = np.asarray(labels).astype(int).ravel()
    edges = np.diff(np.r_[0, y, 0])
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def point_adjust(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Point-adjusted scores: every point of a true segment gets the segment maximum."""
    s = np.asarray(scores, dtype=float).copy()
    for start, end in anomaly_segments(labels):
        s[start:end] = s[start:end].max()
    return s


def evaluate(
    labels: np.ndarray, scores: np.ndarray, with_point_adjust: bool = False
) -> dict[str, float]:
    """Compute ROC-AUC, PR-AUC and best F1; optionally add point-adjusted variants.

    Point-adjusted values are returned under ``pa_`` keys and are optimistic by
    construction.
    """
    f1, threshold = best_f1(labels, scores)
    result = {
        "roc_auc": roc_auc(labels, scores),
        "pr_auc": pr_auc(labels, scores),
        "best_f1": f1,
        "best_f1_threshold": threshold,
        "prevalence": float(np.mean(np.asarray(labels) > 0)),
    }
    if with_point_adjust:
        adjusted = point_adjust(labels, scores)
        result["pa_pr_auc"] = pr_auc(labels, adjusted)
        result["pa_best_f1"] = best_f1(labels, adjusted)[0]
    return result
