"""Maximise the power coefficient with the BEM model directly or with a surrogate."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import differential_evolution

from wtsurrogate.doe import DesignSpace

FREE_VARIABLES: tuple[str, ...] = ("tsr", "pitch_deg", "chord_scale", "twist_scale")


@dataclass(frozen=True)
class OptimumResult:
    """Result of one optimisation run."""

    x: dict[str, float]
    objective_cp: float
    n_evaluations: int
    seconds: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def maximise_cp(
    cp_function: Callable[[np.ndarray], np.ndarray],
    wind_speed: float,
    airfoil_index: int,
    space: DesignSpace | None = None,
    seed: int = 0,
    popsize: int = 12,
    maxiter: int = 60,
) -> OptimumResult:
    """Maximise ``cp_function`` over TSR, pitch, chord scale and twist scale.

    ``cp_function`` takes rows of the raw design matrix. Differential evolution is
    used for both the BEM model and the surrogates so that the comparison uses the
    same optimiser, the same budget rules and one objective call per candidate.
    """
    space = space or DesignSpace()
    bounds = [space.bounds[name] for name in FREE_VARIABLES]
    count = 0

    def objective(z: np.ndarray) -> float:
        nonlocal count
        count += 1
        row = np.array([[wind_speed, z[0], z[1], z[2], z[3], float(airfoil_index)]])
        return -float(cp_function(row)[0])

    start = time.perf_counter()
    result = differential_evolution(
        objective, bounds, seed=seed, popsize=popsize, maxiter=maxiter, tol=1e-6,
        polish=False, init="sobol", updating="immediate",
    )  # fmt: skip
    elapsed = time.perf_counter() - start
    return OptimumResult(
        x=dict(zip(FREE_VARIABLES, (float(v) for v in result.x), strict=True)),
        objective_cp=float(-result.fun),
        n_evaluations=count,
        seconds=elapsed,
    )


def raw_row(optimum: OptimumResult, wind_speed: float, airfoil_index: int) -> np.ndarray:
    """Raw design-matrix row of an optimum, for re-evaluation with another model."""
    x = optimum.x
    return np.array([[wind_speed, x["tsr"], x["pitch_deg"], x["chord_scale"], x["twist_scale"],
                      float(airfoil_index)]])  # fmt: skip


def throughput(
    cp_function: Callable[[np.ndarray], np.ndarray], rows: np.ndarray, repeats: int = 3
) -> dict[str, float]:
    """Measured evaluations per second for single-row calls and for one batched call."""
    single = rows[: min(len(rows), 30)]
    best_single = np.inf
    best_batch = np.inf
    for _ in range(repeats):
        start = time.perf_counter()
        for row in single:
            cp_function(row[None, :])
        best_single = min(best_single, (time.perf_counter() - start) / len(single))
        start = time.perf_counter()
        cp_function(rows)
        best_batch = min(best_batch, (time.perf_counter() - start) / len(rows))
    return {"single_call_per_s": 1.0 / best_single, "batched_per_s": 1.0 / best_batch}
