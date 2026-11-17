"""Short-term forecasting with lag features against persistence baselines.

The forecaster is a direct model: for a horizon of ``h`` steps it learns
``y[t+h]`` from values known at time ``t``. Baselines:

* persistence: ``y_hat[t+h] = y[t]``
* seasonal persistence: ``y_hat[t+h] = y[t+h-season]`` (same time on the previous day)

Evaluation uses a chronological split (no shuffling) and reports MAE, RMSE
and the skill score ``1 - RMSE_model / RMSE_persistence``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


@dataclass(frozen=True)
class FeatureConfig:
    """Lag feature layout."""

    n_lags: int = 8
    season: int = 96  # steps per day (96 at 15 minute resolution)
    horizon: int = 4

    def __post_init__(self) -> None:
        if self.n_lags < 1 or self.horizon < 1:
            raise ValueError("n_lags and horizon must be at least 1")
        if self.horizon > self.season:
            raise ValueError("horizon must not exceed the season length")

    @property
    def min_history(self) -> int:
        """Number of past samples needed (including the current one) to build one row."""
        return max(self.n_lags, self.season - self.horizon + 1, 1)


def _row_features(
    y: np.ndarray, ts: np.ndarray, t_idx: np.ndarray, cfg: FeatureConfig, dt_s: float
) -> np.ndarray:
    """Feature rows for forecast origins ``t_idx`` (targets are at ``t_idx + horizon``)."""
    lags = np.column_stack([y[t_idx - k] for k in range(cfg.n_lags)])
    seasonal = y[t_idx + cfg.horizon - cfg.season]
    roll_mean = lags[:, : min(4, cfg.n_lags)].mean(axis=1)
    trend = lags[:, 0] - lags[:, min(1, cfg.n_lags - 1)]
    target_hour = ((ts[t_idx] + cfg.horizon * dt_s) / 3600.0) % 24.0
    angle = 2.0 * np.pi * target_hour / 24.0
    return np.column_stack([lags, seasonal, roll_mean, trend, np.sin(angle), np.cos(angle)])


def make_supervised(
    y: np.ndarray, ts: np.ndarray, cfg: FeatureConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(X, target, origin_index)`` for every usable forecast origin."""
    y = np.asarray(y, dtype=float)
    ts = np.asarray(ts, dtype=float)
    if len(y) != len(ts):
        raise ValueError("y and ts must have the same length")
    first = cfg.min_history - 1
    last = len(y) - cfg.horizon - 1
    if last < first:
        raise ValueError("series too short for the requested lags and horizon")
    dt_s = float(np.median(np.diff(ts)))
    t_idx = np.arange(first, last + 1)
    return _row_features(y, ts, t_idx, cfg, dt_s), y[t_idx + cfg.horizon], t_idx


class LagForecaster:
    """Gradient boosting on lag, seasonal and time-of-day features."""

    def __init__(self, cfg: FeatureConfig | None = None, seed: int = 0) -> None:
        self.cfg = cfg or FeatureConfig()
        self.model = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.06, max_depth=6, l2_regularization=1.0, random_state=seed
        )
        self._fitted = False

    def fit(self, y: np.ndarray, ts: np.ndarray) -> LagForecaster:
        """Fit on a history series."""
        x, target, _ = make_supervised(y, ts, self.cfg)
        self.model.fit(x, target)
        self._fitted = True
        return self

    def predict_origins(self, y: np.ndarray, ts: np.ndarray, t_idx: np.ndarray) -> np.ndarray:
        """Forecast ``y[t_idx + horizon]`` using data up to each origin only."""
        if not self._fitted:
            raise RuntimeError("forecaster is not fitted")
        y = np.asarray(y, dtype=float)
        ts = np.asarray(ts, dtype=float)
        dt_s = float(np.median(np.diff(ts)))
        pred = self.model.predict(_row_features(y, ts, np.asarray(t_idx), self.cfg, dt_s))
        return np.clip(pred, 0.0, None)

    def predict_next(self, y: np.ndarray, ts: np.ndarray) -> tuple[float, float]:
        """Forecast ``horizon`` steps after the last sample. Returns ``(target_ts, value)``."""
        ts = np.asarray(ts, dtype=float)
        if len(ts) < self.cfg.min_history:
            raise ValueError("not enough history")
        dt_s = float(np.median(np.diff(ts)))
        origin = np.array([len(ts) - 1])
        value = float(self.predict_origins(y, ts, origin)[0])
        return float(ts[-1] + self.cfg.horizon * dt_s), value


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def skill_score(rmse_model: float, rmse_reference: float) -> float:
    """``1 - RMSE_model / RMSE_reference``. Positive means better than the reference."""
    if rmse_reference <= 0.0:
        return 0.0
    return 1.0 - rmse_model / rmse_reference


@dataclass(frozen=True)
class ForecastReport:
    """Hold-out metrics of the model and both baselines."""

    target: str
    horizon_steps: int
    n_train: int
    n_test: int
    mae_model: float
    rmse_model: float
    mae_persistence: float
    rmse_persistence: float
    mae_seasonal: float
    rmse_seasonal: float
    skill_vs_persistence: float
    skill_vs_seasonal: float

    def as_dict(self) -> dict[str, float | int | str]:
        """Plain dictionary for JSON output."""
        return asdict(self)


def evaluate_forecast(
    y: np.ndarray,
    ts: np.ndarray,
    cfg: FeatureConfig | None = None,
    train_fraction: float = 0.75,
    target: str = "series",
    seed: int = 0,
) -> tuple[ForecastReport, LagForecaster]:
    """Chronological train/test evaluation of the lag forecaster and the baselines.

    The model is fitted on the first ``train_fraction`` of the series only.
    Test origins lie entirely in the remaining part, so no test target leaks
    into training.
    """
    cfg = cfg or FeatureConfig()
    y = np.asarray(y, dtype=float)
    ts = np.asarray(ts, dtype=float)
    split = int(len(y) * train_fraction)
    model = LagForecaster(cfg, seed).fit(y[:split], ts[:split])
    _, target_all, t_idx = make_supervised(y, ts, cfg)
    test = t_idx >= split
    origins = t_idx[test]
    if len(origins) == 0:
        raise ValueError("no test samples: series too short")
    y_true = target_all[test]
    pred = model.predict_origins(y, ts, origins)
    persistence = y[origins]
    seasonal = y[origins + cfg.horizon - cfg.season]
    r_model, r_pers, r_seas = rmse(y_true, pred), rmse(y_true, persistence), rmse(y_true, seasonal)
    report = ForecastReport(
        target=target,
        horizon_steps=cfg.horizon,
        n_train=int(np.sum(~test)),
        n_test=int(len(origins)),
        mae_model=mae(y_true, pred),
        rmse_model=r_model,
        mae_persistence=mae(y_true, persistence),
        rmse_persistence=r_pers,
        mae_seasonal=mae(y_true, seasonal),
        rmse_seasonal=r_seas,
        skill_vs_persistence=skill_score(r_model, r_pers),
        skill_vs_seasonal=skill_score(r_model, r_seas),
    )
    return report, model
