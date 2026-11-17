"""Drift monitoring: PSI and Kolmogorov-Smirnov tests on scores and embeddings, with alert rules."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, Field
from scipy import stats

from .embedding import HashingEmbedder

_EPS = 1e-4


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index of a continuous variable.

    Bin edges are the quantiles of the reference sample, so every reference bin
    holds about the same mass. Proportions are floored at a small epsilon to
    keep the logarithm finite. Common reading: below 0.1 stable, 0.1 to 0.25
    moderate shift, above 0.25 significant shift.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if reference.size < bins or current.size == 0:
        raise ValueError("not enough samples for PSI")
    edges = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 2:
        return 0.0 if np.allclose(current, reference[0]) else float("inf")
    inner = edges[1:-1]
    ref_share = np.bincount(
        np.searchsorted(inner, reference, side="right"), minlength=inner.size + 1
    )
    cur_share = np.bincount(np.searchsorted(inner, current, side="right"), minlength=inner.size + 1)
    p = np.clip(ref_share / reference.size, _EPS, None)
    q = np.clip(cur_share / current.size, _EPS, None)
    return float(np.sum((q - p) * np.log(q / p)))


def categorical_psi(reference: list[str], current: list[str]) -> float:
    """PSI over category frequencies (for example predicted labels)."""
    if not reference or not current:
        raise ValueError("not enough samples for PSI")
    categories = sorted(set(reference) | set(current))
    p = np.clip([reference.count(c) / len(reference) for c in categories], _EPS, None)
    q = np.clip([current.count(c) / len(current) for c in categories], _EPS, None)
    return float(np.sum((q - p) * np.log(q / p)))


def ks_test(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test. Returns ``(statistic, p_value)``."""
    result = stats.ks_2samp(np.asarray(reference, dtype=float), np.asarray(current, dtype=float))
    return float(result.statistic), float(result.pvalue)


def embedding_drift(
    reference: np.ndarray, current: np.ndarray, n_components: int = 8, alpha: float = 0.05
) -> dict[str, float]:
    """Summarise drift between two sets of embeddings.

    - ``centroid_cosine_distance``: 1 minus the cosine similarity of the mean vectors.
    - ``drifted_component_share``: share of the top principal components (fitted on
      the reference) whose projections differ by a KS test at level ``alpha`` with
      Bonferroni correction.
    - ``max_component_ks``: largest KS statistic over those components.
    - ``centroid_distance_psi``: PSI of each point's distance to the reference centroid.
    """
    if reference.ndim != 2 or current.ndim != 2 or reference.shape[1] != current.shape[1]:
        raise ValueError("embeddings must be 2-D arrays with the same width")
    mean_ref, mean_cur = reference.mean(axis=0), current.mean(axis=0)
    denominator = float(np.linalg.norm(mean_ref) * np.linalg.norm(mean_cur))
    cosine = float(mean_ref @ mean_cur) / denominator if denominator else 0.0
    k = min(n_components, reference.shape[1], reference.shape[0] - 1)
    _, _, components = np.linalg.svd(reference - mean_ref, full_matrices=False)
    ref_proj = (reference - mean_ref) @ components[:k].T
    cur_proj = (current - mean_ref) @ components[:k].T
    results = [ks_test(ref_proj[:, i], cur_proj[:, i]) for i in range(k)]
    drifted = sum(p_value < alpha / k for _, p_value in results)
    ref_dist = np.linalg.norm(reference - mean_ref, axis=1)
    cur_dist = np.linalg.norm(current - mean_ref, axis=1)
    return {
        "centroid_cosine_distance": round(1.0 - cosine, 6),
        "drifted_component_share": round(drifted / k, 4),
        "max_component_ks": round(max(s for s, _ in results), 4),
        "centroid_distance_psi": round(psi(ref_dist, cur_dist), 4),
    }


class AlertRule(BaseModel):
    """Thresholds for one drift metric. ``direction`` says which side is bad."""

    metric: str
    warn: float
    critical: float
    direction: Literal["above", "below"] = "above"


class Alert(BaseModel):
    metric: str
    severity: Literal["warning", "critical"]
    value: float
    threshold: float
    message: str


class DriftReport(BaseModel):
    n_reference: int
    n_current: int
    metrics: dict[str, float]
    alerts: list[Alert] = Field(default_factory=list)

    @property
    def status(self) -> str:
        severities = {a.severity for a in self.alerts}
        return "critical" if "critical" in severities else "warning" if severities else "ok"


def load_alert_rules(path: str | Path) -> list[AlertRule]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [AlertRule.model_validate(r) for r in raw["alerts"]]


def evaluate_alerts(metrics: dict[str, float], rules: list[AlertRule]) -> list[Alert]:
    """Turn metric values into alerts. Metrics without a rule are ignored."""
    alerts = []
    for rule in rules:
        value = metrics.get(rule.metric)
        if value is None:
            continue
        sign = 1.0 if rule.direction == "above" else -1.0
        for severity, threshold in (("critical", rule.critical), ("warning", rule.warn)):
            if sign * value > sign * threshold:
                alerts.append(
                    Alert(
                        metric=rule.metric,
                        severity=severity,  # type: ignore[arg-type]
                        value=value,
                        threshold=threshold,
                        message=f"{rule.metric}={value:g} is {rule.direction} {threshold:g}",
                    )
                )
                break
    return alerts


def monitor(
    reference: list[dict[str, Any]],
    current: list[dict[str, Any]],
    rules: list[AlertRule],
    embed_dim: int = 256,
) -> DriftReport:
    """Compare two windows of inference logs.

    Each log record needs ``text``, ``score`` (model confidence) and ``label``
    (predicted label). ``latency_ms`` is used when present in both windows.
    """
    if len(reference) < 30 or len(current) < 30:
        raise ValueError("need at least 30 records in each window")
    embedder = HashingEmbedder(dim=embed_dim)
    ref_scores = np.asarray([r["score"] for r in reference], dtype=float)
    cur_scores = np.asarray([r["score"] for r in current], dtype=float)
    ks_stat, ks_p = ks_test(ref_scores, cur_scores)
    metrics: dict[str, float] = {
        "score_psi": round(psi(ref_scores, cur_scores), 4),
        "score_ks_statistic": round(ks_stat, 4),
        "score_ks_pvalue": float(f"{ks_p:.3g}"),
        "mean_score_drop": round(float(ref_scores.mean() - cur_scores.mean()), 4),
        "label_psi": round(
            categorical_psi([r["label"] for r in reference], [r["label"] for r in current]), 4
        ),
    }
    metrics.update(
        embedding_drift(
            embedder.embed([r["text"] for r in reference]),
            embedder.embed([r["text"] for r in current]),
        )
    )
    if all("latency_ms" in r for r in reference) and all("latency_ms" in r for r in current):
        ref_p95 = float(np.percentile([r["latency_ms"] for r in reference], 95))
        cur_p95 = float(np.percentile([r["latency_ms"] for r in current], 95))
        metrics["latency_p95_ratio"] = round(cur_p95 / ref_p95, 4) if ref_p95 > 0 else 1.0
    return DriftReport(
        n_reference=len(reference),
        n_current=len(current),
        metrics=metrics,
        alerts=evaluate_alerts(metrics, rules),
    )
