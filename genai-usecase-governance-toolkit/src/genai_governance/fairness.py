"""Responsible AI evaluation on a tabular prediction file: performance and group fairness metrics."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class PredictionData:
    """Binary labels and predictions with one sensitive attribute per row."""

    y_true: np.ndarray
    y_pred: np.ndarray
    group: np.ndarray
    y_score: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = len(self.y_true)
        if n == 0:
            raise ValueError("prediction data is empty")
        if len(self.y_pred) != n or len(self.group) != n:
            raise ValueError("y_true, y_pred and group must have the same length")
        for name, arr in (("y_true", self.y_true), ("y_pred", self.y_pred)):
            if not np.isin(arr, (0, 1)).all():
                raise ValueError(f"{name} must be binary (0/1)")


def load_predictions(
    path: str | Path,
    y_true: str = "y_true",
    y_pred: str = "y_pred",
    group: str = "group",
    y_score: str | None = "y_score",
) -> PredictionData:
    """Read a CSV file. The score column is optional and skipped when absent."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"no rows in {path}")
    for col in (y_true, y_pred, group):
        if col not in rows[0]:
            raise ValueError(f"column '{col}' not found in {path}; available: {sorted(rows[0])}")
    score = None
    if y_score and y_score in rows[0]:
        score = np.array([float(r[y_score]) for r in rows])
    return PredictionData(
        y_true=np.array([int(float(r[y_true])) for r in rows]),
        y_pred=np.array([int(float(r[y_pred])) for r in rows]),
        group=np.array([r[group] for r in rows]),
        y_score=score,
    )


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    """Return (tp, fp, tn, fn) for binary arrays."""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, fp, tn, fn


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the ROC curve via the rank-sum (Mann-Whitney U) identity, with tie handling."""
    pos, neg = int((y_true == 1).sum()), int((y_true == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    ranks = np.empty(len(y_score))
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1  # average rank for ties
        i = j + 1
    return float((ranks[y_true == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


@dataclass
class PerformanceMetrics:
    """Overall binary classification metrics."""

    n: int
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    balanced_accuracy: float
    tp: int
    fp: int
    tn: int
    fn: int
    roc_auc: float | None = None
    brier: float | None = None


def performance_metrics(data: PredictionData) -> PerformanceMetrics:
    """Compute overall performance. AUC and Brier score need the score column."""
    tp, fp, tn, fn = confusion(data.y_true, data.y_pred)
    precision, recall = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    auc = brier = None
    if data.y_score is not None:
        auc = roc_auc(data.y_true, data.y_score)
        brier = float(np.mean((data.y_score - data.y_true) ** 2))
    return PerformanceMetrics(
        n=len(data.y_true),
        accuracy=_safe_div(tp + tn, tp + fp + tn + fn),
        precision=precision,
        recall=recall,
        specificity=specificity,
        f1=_safe_div(2 * precision * recall, precision + recall),
        balanced_accuracy=(recall + specificity) / 2,
        tp=tp, fp=fp, tn=tn, fn=fn,
        roc_auc=auc,
        brier=brier,
    )


@dataclass
class GroupMetrics:
    """Rates for one group of the sensitive attribute."""

    group: str
    n: int
    selection_rate: float
    tpr: float
    fpr: float
    accuracy: float


@dataclass
class FairnessReport:
    """Group fairness metrics with thresholds and resulting flags."""

    groups: list[GroupMetrics]
    demographic_parity_difference: float
    equalised_odds_difference: float
    tpr_difference: float
    fpr_difference: float
    disparate_impact: float
    flags: list[str] = field(default_factory=list)
    dpd_ci95: tuple[float, float] | None = None


def group_metrics(data: PredictionData) -> list[GroupMetrics]:
    """Per-group selection rate, true positive rate, false positive rate and accuracy."""
    out = []
    for g in sorted(set(data.group.tolist())):
        mask = data.group == g
        yt, yp = data.y_true[mask], data.y_pred[mask]
        tp, fp, tn, fn = confusion(yt, yp)
        out.append(
            GroupMetrics(
                group=str(g),
                n=int(mask.sum()),
                selection_rate=float(yp.mean()),
                tpr=_safe_div(tp, tp + fn),
                fpr=_safe_div(fp, fp + tn),
                accuracy=_safe_div(tp + tn, len(yt)),
            )
        )
    return out


def _spread(values: list[float]) -> float:
    vals = [v for v in values if not np.isnan(v)]
    return float(max(vals) - min(vals)) if vals else float("nan")


def _dpd(y_pred: np.ndarray, group: np.ndarray) -> float:
    rates = [float(y_pred[group == g].mean()) for g in np.unique(group)]
    return max(rates) - min(rates)


def fairness_report(
    data: PredictionData,
    dpd_threshold: float = 0.10,
    eod_threshold: float = 0.10,
    di_threshold: float = 0.80,
    min_group_size: int = 30,
    bootstrap: int = 0,
    seed: int = 42,
) -> FairnessReport:
    """Compute group fairness metrics over all groups of the sensitive attribute.

    Demographic parity difference: max minus min selection rate across groups.
    Equalised odds difference: the larger of the TPR spread and the FPR spread.
    Disparate impact: min divided by max selection rate (the four-fifths rule
    flags values below 0.8). With ``bootstrap`` > 0 a percentile 95 percent
    confidence interval for the demographic parity difference is added.
    """
    groups = group_metrics(data)
    if len(groups) < 2:
        raise ValueError("at least two groups are required for fairness metrics")
    rates = [g.selection_rate for g in groups]
    tpr_diff, fpr_diff = _spread([g.tpr for g in groups]), _spread([g.fpr for g in groups])
    dpd = max(rates) - min(rates)
    eod = float(np.nanmax([tpr_diff, fpr_diff]))
    di = _safe_div(min(rates), max(rates))

    flags = []
    if dpd > dpd_threshold:
        flags.append(f"demographic parity difference {dpd:.3f} exceeds threshold {dpd_threshold:.2f}")
    if eod > eod_threshold:
        flags.append(f"equalised odds difference {eod:.3f} exceeds threshold {eod_threshold:.2f}")
    if not np.isnan(di) and di < di_threshold:
        flags.append(f"disparate impact ratio {di:.3f} is below threshold {di_threshold:.2f}")
    for g in groups:
        if g.n < min_group_size:
            flags.append(f"group '{g.group}' has only {g.n} rows, its rates are unreliable")

    ci = None
    if bootstrap > 0:
        rng = np.random.default_rng(seed)
        n = len(data.y_pred)
        stats = []
        for _ in range(bootstrap):
            idx = rng.integers(0, n, size=n)
            if len(np.unique(data.group[idx])) < 2:
                continue
            stats.append(_dpd(data.y_pred[idx], data.group[idx]))
        if stats:
            ci = (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))
    return FairnessReport(
        groups=groups,
        demographic_parity_difference=dpd,
        equalised_odds_difference=eod,
        tpr_difference=tpr_diff,
        fpr_difference=fpr_diff,
        disparate_impact=di,
        flags=flags,
        dpd_ci95=ci,
    )
