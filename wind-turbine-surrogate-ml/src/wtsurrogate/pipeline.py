"""End-to-end workflow: DoE, BEM dataset, surrogates, uncertainty, optimisation, AEP.

``run_pipeline`` returns one JSON-serialisable dictionary with every number that
appears in the documentation, plus a ``PipelineArtifacts`` object holding fitted
models and arrays for plotting.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning

from wtsurrogate.dataset import BemOracle, Dataset, FeatureBuilder, generate_dataset
from wtsurrogate.doe import RAW_COLUMNS, VARIABLES, DesignSpace
from wtsurrogate.interpret import partial_dependence, permutation_importance
from wtsurrogate.models import (
    MODEL_NAMES,
    ModelReport,
    Surrogate,
    model_specs,
    regression_metrics,
    tune_and_evaluate,
)
from wtsurrogate.optimise import maximise_cp, raw_row, throughput
from wtsurrogate.power import (
    PowerCurve,
    TurbineLimits,
    annual_energy_production,
    capacity_factor,
    compute_power_curve,
    pitch_for_rated_power,
)
from wtsurrogate.uncertainty import (
    QuantileBoosting,
    conformal_half_width,
    forest_predict_std,
    gaussian_calibration,
    gaussian_interval,
    gp_predict_std,
    interval_quality,
)


@dataclass
class PipelineArtifacts:
    """Objects kept in memory for plotting and further analysis."""

    dataset: Dataset
    oracle: BemOracle
    features: FeatureBuilder
    reports: dict[str, ModelReport]
    surrogates: dict[str, Surrogate]
    best_name: str
    splits: dict[str, np.ndarray]
    predictions: dict[str, np.ndarray] = field(default_factory=dict)
    intervals: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    pdp: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    power_curves: dict[str, PowerCurve] = field(default_factory=dict)


def split_indices(n: int, seed: int, fractions: tuple[float, float] = (0.7, 0.15)) -> dict:
    """Random train / calibration / test split of ``n`` rows."""
    order = np.random.default_rng(seed).permutation(n)
    n_train = int(round(fractions[0] * n))
    n_cal = int(round(fractions[1] * n))
    return {
        "train": order[:n_train],
        "calibration": order[n_train : n_train + n_cal],
        "test": order[n_train + n_cal :],
    }


def _uncertainty_study(
    reports: dict[str, ModelReport], x: dict[str, np.ndarray], y: dict[str, np.ndarray],
    level: float, artifacts: PipelineArtifacts,
) -> dict[str, Any]:  # fmt: skip
    """Native intervals per model family and split conformal intervals for all of them."""
    out: dict[str, Any] = {"nominal_level": level}
    native: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if "gaussian_process" in reports:
        mean, std = gp_predict_std(reports["gaussian_process"].estimator, x["test"])
        native["gaussian_process"] = gaussian_interval(mean, std, level)
        out["gaussian_process_calibration"] = gaussian_calibration(y["test"], mean, std)
    if "random_forest" in reports:
        mean, std = forest_predict_std(reports["random_forest"].estimator, x["test"])
        native["random_forest"] = gaussian_interval(mean, std, level)
    if "gradient_boosting" in reports:
        quantile = QuantileBoosting(reports["gradient_boosting"].estimator, level)
        native["gradient_boosting"] = quantile.fit(x["train"], y["train"]).interval(x["test"])
    if "polynomial_ridge" in reports:
        rep = reports["polynomial_ridge"]
        mean = rep.estimator.predict(x["test"])
        native["polynomial_ridge"] = gaussian_interval(mean, np.full_like(mean, rep.cv_rmse_mean),
                                                       level)  # fmt: skip
    methods = {
        "gaussian_process": "GP posterior standard deviation",
        "random_forest": "spread of individual trees",
        "gradient_boosting": "quantile regression (5 and 95 percent)",
        "polynomial_ridge": "constant sigma equal to CV RMSE",
    }
    rows = {}
    for name, rep in reports.items():
        lo, hi = native[name]
        quality = interval_quality(y["test"], lo, hi)
        residual = y["calibration"] - rep.estimator.predict(x["calibration"])
        half = conformal_half_width(residual, level)
        pred = artifacts.predictions[name]
        conformal = interval_quality(y["test"], pred - half, pred + half)
        rows[name] = {
            "native_method": methods[name],
            "native_coverage": quality["coverage"],
            "native_mean_width": quality["mean_width"],
            "conformal_coverage": conformal["coverage"],
            "conformal_mean_width": conformal["mean_width"],
        }
        artifacts.intervals[name] = (lo, hi)
    out["models"] = rows
    return out


def _optimisation_study(
    oracle: BemOracle, surrogates: dict[str, Surrogate], best_name: str, wind_speed: float,
    seed: int, maxiter: int, setup_seconds: dict[str, float],
) -> dict[str, Any]:  # fmt: skip
    """Compare the optimum found on each surrogate with the direct BEM optimum."""
    out: dict[str, Any] = {"wind_speed": wind_speed, "airfoils": {}}
    for index, airfoil in enumerate(oracle.space.airfoils):
        direct = maximise_cp(oracle.cp, wind_speed, index, oracle.space, seed, maxiter=maxiter)
        names = list(surrogates) if index == 0 else [best_name]
        rows = {}
        for name in names:
            found = maximise_cp(surrogates[name].predict, wind_speed, index, oracle.space, seed,
                                maxiter=maxiter)  # fmt: skip
            verified = float(oracle.cp(raw_row(found, wind_speed, index))[0])
            rows[name] = {
                **found.as_dict(),
                "cp_bem_at_surrogate_optimum": verified,
                "cp_gap_to_direct": direct.objective_cp - verified,
                "surrogate_overprediction": found.objective_cp - verified,
                "optimiser_speed_up": direct.seconds / found.seconds,
                "speed_up_including_setup": direct.seconds
                / (found.seconds + setup_seconds[name]),
            }
        out["airfoils"][airfoil] = {"direct_bem": direct.as_dict(), "surrogates": rows}
    return out


def _power_study(
    oracle: BemOracle, surrogate: Surrogate, limits: TurbineLimits,
    mean_speeds: tuple[float, ...], artifacts: PipelineArtifacts,
) -> dict[str, Any]:  # fmt: skip
    """Power curves and AEP from the BEM model and from the surrogate."""
    out: dict[str, Any] = {"limits": limits.__dict__, "weibull_shape": 2.0, "airfoils": {}}
    upper = oracle.space.bounds["wind_speed"][1]
    for index, airfoil in enumerate(oracle.space.airfoils):
        curves = {
            "bem": compute_power_curve(oracle.cp, index, limits, max_model_speed=upper),
            "surrogate": compute_power_curve(surrogate.predict, index, limits,
                                             max_model_speed=upper),
        }  # fmt: skip
        rows = {}
        for mean in mean_speeds:
            aep = {k: annual_energy_production(c.speed, c.power, mean) for k, c in curves.items()}
            rows[f"{mean:g}"] = {
                "aep_bem_gwh": aep["bem"] / 1e9,
                "aep_surrogate_gwh": aep["surrogate"] / 1e9,
                "relative_error_percent": 100.0 * (aep["surrogate"] - aep["bem"]) / aep["bem"],
                "capacity_factor_bem": capacity_factor(aep["bem"], limits.rated_power),
            }
        below = curves["bem"].speed <= upper
        diff = (curves["surrogate"].power - curves["bem"].power)[below]
        out["airfoils"][airfoil] = {
            "rated_speed_bem": curves["bem"].rated_speed(limits.rated_power),
            "rated_speed_surrogate": curves["surrogate"].rated_speed(limits.rated_power),
            "max_cp_bem": float(np.nanmax(curves["bem"].cp)),
            "max_power_curve_error_kw": float(np.max(np.abs(diff)) / 1e3),
            "pitch_for_rated_deg_bem": {
                f"{u:g}": pitch_for_rated_power(oracle.rotors[index], u, limits)
                for u in (15.0, 20.0, 25.0)
            },
            "aep_by_mean_wind_speed": rows,
        }
        artifacts.power_curves[f"{airfoil}/bem"] = curves["bem"]
        artifacts.power_curves[f"{airfoil}/surrogate"] = curves["surrogate"]
    return out


def run_pipeline(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], PipelineArtifacts]:
    """Run the full workflow and return ``(results, artifacts)``.

    Accepts the arguments of ``_run_pipeline``. Gaussian process length-scale bound
    warnings are silenced: inactive inputs legitimately run into the upper bound.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return _run_pipeline(*args, **kwargs)


def _run_pipeline(
    n_samples: int = 1500,
    seed: int = 0,
    budget: float = 1.0,
    models: tuple[str, ...] = MODEL_NAMES,
    optimiser_maxiter: int = 40,
    verbose: bool = True,
) -> tuple[dict[str, Any], PipelineArtifacts]:
    """Implementation of ``run_pipeline``.

    ``budget`` scales the hyperparameter search, ``optimiser_maxiter`` bounds the
    differential evolution generations of the optimisation study.
    """

    def log(message: str) -> None:
        if verbose:
            print(message, flush=True)

    space = DesignSpace()
    oracle = BemOracle(space)
    features = FeatureBuilder(oracle.rotors)
    dataset = generate_dataset(n_samples, seed, oracle=oracle)
    log(f"dataset: {len(dataset)} of {n_samples} BEM runs kept ({dataset.n_failed} failed), "
        f"{dataset.bem_seconds:.2f} s")  # fmt: skip
    splits = split_indices(len(dataset), seed)
    feats = features.transform(dataset.raw)
    x = {k: feats[idx] for k, idx in splits.items()}
    y = {k: dataset.cp[idx] for k, idx in splits.items()}

    specs = model_specs(feats.shape[1], seed, budget)
    reports: dict[str, ModelReport] = {}
    for name in models:
        reports[name] = tune_and_evaluate(specs[name], x["train"], y["train"], x["test"],
                                          y["test"], seed=seed)  # fmt: skip
        rep = reports[name]
        log(f"{name:<18} cv rmse {rep.cv_rmse_mean:.5f} +/- {rep.cv_rmse_std:.5f}  "
            f"test rmse {rep.test['rmse']:.5f}  r2 {rep.test['r2']:.4f}  "
            f"search {rep.search_seconds:.1f} s")  # fmt: skip
    best_name = min(reports, key=lambda k: reports[k].cv_rmse_mean)
    surrogates = {k: Surrogate(r.estimator, features, k) for k, r in reports.items()}
    artifacts = PipelineArtifacts(dataset, oracle, features, reports, surrogates, best_name, splits)
    artifacts.predictions = {k: r.estimator.predict(x["test"]) for k, r in reports.items()}
    best = surrogates[best_name]
    log(f"selected by CV RMSE: {best_name}")

    ct_model = clone(reports[best_name].estimator)
    spec = specs[best_name]
    rows = splits["train"][: spec.max_train] if spec.max_train else splits["train"]
    ct_model.fit(feats[rows], dataset.ct[rows])
    ct_metrics = regression_metrics(dataset.ct[splits["test"]], ct_model.predict(x["test"]))

    sizes = [s for s in (50, 100, 200, 400, 800) if s < len(rows)] + [len(rows)]
    learning_curve = {}
    for size in sizes:
        model = clone(reports[best_name].estimator).fit(x["train"][:size], y["train"][:size])
        learning_curve[str(size)] = regression_metrics(y["test"], model.predict(x["test"]))["rmse"]

    uncertainty = _uncertainty_study(reports, x, y, 0.9, artifacts)
    raw_test = dataset.raw[splits["test"]]
    importance = permutation_importance(best.predict, raw_test, y["test"], seed=seed)
    for variable in VARIABLES:
        grid = np.linspace(*space.bounds[variable], 25)
        artifacts.pdp[variable] = {
            "grid": grid,
            "surrogate": partial_dependence(best.predict, raw_test, variable, grid),
            "bem": partial_dependence(oracle.cp, raw_test, variable, grid),
        }
    pdp_error = {
        v: float(np.max(np.abs(p["surrogate"] - p["bem"]))) for v, p in artifacts.pdp.items()
    }
    log("uncertainty, importance and partial dependence done")

    bench_rows = raw_test[:200]
    speed = {"bem": throughput(oracle.cp, bench_rows)}
    speed.update({k: throughput(s.predict, bench_rows) for k, s in surrogates.items()})
    setup = {k: dataset.bem_seconds + r.search_seconds for k, r in reports.items()}
    start = time.perf_counter()
    optimisation = _optimisation_study(oracle, surrogates, best_name, 9.0, seed,
                                       optimiser_maxiter, setup)  # fmt: skip
    log(f"optimisation study done in {time.perf_counter() - start:.1f} s")
    power = _power_study(oracle, best, TurbineLimits(radius=oracle.rotors[0].radius),
                         (7.5, 8.5, 10.0), artifacts)  # fmt: skip
    log("power curve and AEP done")

    results: dict[str, Any] = {
        "meta": {
            "data": "generated by the BEM solver in this repository (not CFD, not measured)",
            "seed": seed,
            "n_requested": n_samples,
            "n_kept": len(dataset),
            "n_failed_bem": dataset.n_failed,
            "bem_seconds": dataset.bem_seconds,
            "split_sizes": {k: int(len(v)) for k, v in splits.items()},
            "raw_columns": list(RAW_COLUMNS),
            "feature_names": list(features.names),
            "bounds": {k: list(v) for k, v in space.bounds.items()},
            "airfoils": list(space.airfoils),
            "cp_range": [float(dataset.cp.min()), float(dataset.cp.max())],
            "search_budget": budget,
        },
        "models": {k: r.summary() for k, r in reports.items()},
        "best_model": best_name,
        "ct_model_test": ct_metrics,
        "learning_curve_test_rmse": learning_curve,
        "uncertainty": uncertainty,
        "permutation_importance": importance,
        "partial_dependence_max_abs_error": pdp_error,
        "throughput_evals_per_s": speed,
        "optimisation": optimisation,
        "power": power,
    }
    return results, artifacts
