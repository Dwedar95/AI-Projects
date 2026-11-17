"""Predictive uncertainty for the surrogates and checks of its calibration.

Four native interval estimators are implemented, one per model family, plus split
conformal prediction as a model-agnostic wrapper with a finite-sample coverage
guarantee under exchangeability (Vovk et al., 2005; Lei et al., 2018).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import norm
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline


def gaussian_interval(
    mean: np.ndarray, std: np.ndarray, level: float
) -> tuple[np.ndarray, np.ndarray]:
    """Central interval of a normal distribution with the given coverage ``level``."""
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    z = norm.ppf(0.5 + level / 2.0)
    return mean - z * std, mean + z * std


def gp_predict_std(pipeline: Pipeline, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Posterior mean and standard deviation of a scaler + GP pipeline."""
    scaled = pipeline[:-1].transform(x)
    mean, std = pipeline[-1].predict(scaled, return_std=True)
    return mean, std


def forest_predict_std(
    forest: RandomForestRegressor, x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Forest mean and the standard deviation of the individual tree predictions.

    The tree spread measures disagreement between trees, not predictive noise, and
    is usually not calibrated as a prediction interval.
    """
    trees = np.stack([tree.predict(x) for tree in forest.estimators_])
    return trees.mean(axis=0), trees.std(axis=0)


class QuantileBoosting:
    """Lower and upper quantile models with the hyperparameters of a tuned booster."""

    def __init__(self, tuned: HistGradientBoostingRegressor, level: float = 0.9) -> None:
        params: dict[str, Any] = tuned.get_params()
        self.level = level
        tail = (1.0 - level) / 2.0
        self.lower = clone(tuned).set_params(**{**params, "loss": "quantile", "quantile": tail})
        self.upper = clone(tuned).set_params(
            **{**params, "loss": "quantile", "quantile": 1.0 - tail}
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> QuantileBoosting:
        self.lower.fit(x, y)
        self.upper.fit(x, y)
        return self

    def interval(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = self.lower.predict(x), self.upper.predict(x)
        return np.minimum(lo, hi), np.maximum(lo, hi)


def conformal_half_width(calibration_residuals: np.ndarray, level: float) -> float:
    """Split conformal half-width from absolute residuals on a calibration set.

    Uses the ``ceil((n + 1) * level) / n`` empirical quantile, which gives marginal
    coverage of at least ``level`` for exchangeable data.
    """
    res = np.sort(np.abs(np.asarray(calibration_residuals, dtype=float)))
    n = res.size
    if n == 0:
        raise ValueError("calibration set is empty")
    rank = int(np.ceil((n + 1) * level))
    return float(res[min(rank, n) - 1])


def interval_quality(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict[str, float]:
    """Empirical coverage and mean width of prediction intervals."""
    inside = (y >= lo) & (y <= hi)
    return {"coverage": float(np.mean(inside)), "mean_width": float(np.mean(hi - lo))}


def gaussian_calibration(
    y: np.ndarray, mean: np.ndarray, std: np.ndarray,
    levels: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95),
) -> dict[str, float]:  # fmt: skip
    """Empirical coverage of Gaussian intervals at several nominal levels."""
    out = {}
    for level in levels:
        lo, hi = gaussian_interval(mean, np.maximum(std, 1e-12), level)
        out[f"{level:.2f}"] = interval_quality(y, lo, hi)["coverage"]
    return out
