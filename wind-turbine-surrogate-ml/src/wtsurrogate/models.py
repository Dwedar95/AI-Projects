"""Surrogate model zoo: definition, hyperparameter search and evaluation."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from wtsurrogate.dataset import FeatureBuilder

MODEL_NAMES: tuple[str, ...] = ("gradient_boosting", "random_forest", "gaussian_process",
                                "polynomial_ridge")  # fmt: skip


@dataclass(frozen=True)
class ModelSpec:
    """An estimator with its hyperparameter search space."""

    name: str
    estimator: BaseEstimator
    param_space: dict[str, list[Any]]
    n_iter: int
    max_train: int | None = None


def _gp_kernel(n_features: int, nu: float) -> Any:
    return ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
        length_scale=np.ones(n_features), length_scale_bounds=(5e-2, 5e2), nu=nu
    ) + WhiteKernel(1e-4, (1e-8, 1e-1))


def model_specs(n_features: int, seed: int = 0, budget: float = 1.0) -> dict[str, ModelSpec]:
    """Return the four surrogate families with their search spaces.

    ``budget`` scales the number of random-search iterations (1.0 is the setting
    used for the documented results, smaller values are used by the tests).
    The Gaussian process is trained on at most ``max_train`` points because its
    cost grows cubically with the number of samples.
    """
    n_iter = lambda full: max(2, int(round(full * budget)))  # noqa: E731
    return {
        "gradient_boosting": ModelSpec(
            "gradient_boosting",
            HistGradientBoostingRegressor(random_state=seed, early_stopping=False),
            {
                "learning_rate": [0.03, 0.05, 0.1, 0.2],
                "max_iter": [200, 400, 800],
                "max_leaf_nodes": [7, 15, 31],
                "min_samples_leaf": [3, 5, 10, 20],
                "l2_regularization": [0.0, 0.1, 1.0],
            },
            n_iter(16),
        ),
        "random_forest": ModelSpec(
            "random_forest",
            RandomForestRegressor(random_state=seed, n_jobs=1),
            {
                "n_estimators": [150, 300],
                "max_features": [0.4, 0.6, 0.8, 1.0],
                "min_samples_leaf": [1, 2, 4],
                "max_depth": [None, 12, 20],
            },
            n_iter(8),
        ),
        "gaussian_process": ModelSpec(
            "gaussian_process",
            Pipeline([
                ("scale", StandardScaler()),
                ("gp", GaussianProcessRegressor(_gp_kernel(n_features, 2.5), normalize_y=True,
                                                random_state=seed)),
            ]),
            {"gp__kernel": [_gp_kernel(n_features, 1.5), _gp_kernel(n_features, 2.5)]},
            2,
            max_train=500,
        ),
        "polynomial_ridge": ModelSpec(
            "polynomial_ridge",
            Pipeline([
                ("scale", StandardScaler()),
                ("poly", PolynomialFeatures(include_bias=False)),
                ("scale2", StandardScaler()),
                ("ridge", Ridge()),
            ]),
            {"poly__degree": [2, 3, 4], "ridge__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]},
            n_iter(10),
        ),
    }  # fmt: skip


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """RMSE, MAE, maximum absolute error and R2."""
    err = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "max_abs_error": float(np.max(np.abs(err))),
        "r2": float(1.0 - np.sum(err**2) / ss_tot) if ss_tot > 0 else float("nan"),
    }


@dataclass
class ModelReport:
    """Outcome of tuning and evaluating one surrogate family."""

    name: str
    estimator: BaseEstimator
    best_params: dict[str, str]
    n_train_used: int
    cv_rmse_mean: float
    cv_rmse_std: float
    test: dict[str, float]
    search_seconds: float
    fit_seconds: float
    predict_us_per_sample: float
    extra: dict[str, float] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """JSON-serialisable summary without the fitted estimator."""
        return {k: v for k, v in self.__dict__.items() if k != "estimator"}


def tune_and_evaluate(
    spec: ModelSpec,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    cv: int = 5,
    seed: int = 0,
) -> ModelReport:
    """Random search with K-fold cross-validation on the training split.

    The test split is only touched once, after the search has selected and refitted
    the best configuration.
    """
    rng = np.random.default_rng(seed)
    if spec.max_train is not None and len(x_train) > spec.max_train:
        rows = rng.choice(len(x_train), size=spec.max_train, replace=False)
        x_fit, y_fit = x_train[rows], y_train[rows]
    else:
        x_fit, y_fit = x_train, y_train
    n_candidates = int(np.prod([len(v) for v in spec.param_space.values()]))
    search = RandomizedSearchCV(
        clone(spec.estimator),
        spec.param_space,
        n_iter=min(spec.n_iter, n_candidates),
        scoring="neg_root_mean_squared_error",
        cv=KFold(cv, shuffle=True, random_state=seed),
        random_state=seed,
        refit=True,
        n_jobs=1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        start = time.perf_counter()
        search.fit(x_fit, y_fit)
        search_seconds = time.perf_counter() - start
    best = search.best_index_
    start = time.perf_counter()
    y_pred = search.best_estimator_.predict(x_test)
    predict_us = (time.perf_counter() - start) / len(x_test) * 1e6
    return ModelReport(
        name=spec.name,
        estimator=search.best_estimator_,
        best_params={k: str(v) for k, v in search.best_params_.items()},
        n_train_used=len(x_fit),
        cv_rmse_mean=float(-search.cv_results_["mean_test_score"][best]),
        cv_rmse_std=float(search.cv_results_["std_test_score"][best]),
        test=regression_metrics(y_test, y_pred),
        search_seconds=search_seconds,
        fit_seconds=float(search.refit_time_),
        predict_us_per_sample=predict_us,
    )


class Surrogate:
    """A fitted estimator together with its feature builder.

    ``predict`` takes rows of the raw design matrix, so the surrogate has the same
    call signature as ``BemOracle.cp``.
    """

    def __init__(self, estimator: BaseEstimator, features: FeatureBuilder, name: str = "") -> None:
        self.estimator = estimator
        self.features = features
        self.name = name
        self.n_calls = 0

    def predict(self, raw: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(raw, dtype=float))
        self.n_calls += len(x)
        return self.estimator.predict(self.features.transform(x))
