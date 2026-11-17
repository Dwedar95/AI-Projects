"""Model interpretation on the level of the raw design variables.

Engineered features are functions of the design variables, so permuting or
sweeping a single engineered column would create physically inconsistent inputs.
Both tools below therefore perturb the raw variable and let the prediction
function rebuild the features.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from wtsurrogate.doe import RAW_COLUMNS

PredictFn = Callable[[np.ndarray], np.ndarray]


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    return float(1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


def permutation_importance(
    predict: PredictFn, raw: np.ndarray, y: np.ndarray, n_repeats: int = 10, seed: int = 0
) -> dict[str, dict[str, float]]:
    """Drop in R2 when one raw design variable is shuffled (Breiman, 2001)."""
    rng = np.random.default_rng(seed)
    base = _r2(y, predict(raw))
    out: dict[str, dict[str, float]] = {}
    for col, name in enumerate(RAW_COLUMNS):
        drops = []
        for _ in range(n_repeats):
            shuffled = raw.copy()
            shuffled[:, col] = rng.permutation(shuffled[:, col])
            drops.append(base - _r2(y, predict(shuffled)))
        out[name] = {"mean": float(np.mean(drops)), "std": float(np.std(drops))}
    return out


def partial_dependence(
    predict: PredictFn, raw: np.ndarray, variable: str, grid: np.ndarray, max_rows: int = 300
) -> np.ndarray:
    """Average prediction when ``variable`` is forced to each value of ``grid``."""
    col = RAW_COLUMNS.index(variable)
    background = raw[:max_rows]
    values = np.empty(len(grid))
    for i, value in enumerate(grid):
        forced = background.copy()
        forced[:, col] = value
        values[i] = float(np.mean(predict(forced)))
    return values
