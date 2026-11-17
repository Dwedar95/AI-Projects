"""Design of experiments: Latin hypercube sampling over the design space."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wtsurrogate.airfoils import AIRFOIL_NAMES

VARIABLES: tuple[str, ...] = ("wind_speed", "tsr", "pitch_deg", "chord_scale", "twist_scale")
RAW_COLUMNS: tuple[str, ...] = (*VARIABLES, "airfoil_index")


@dataclass(frozen=True)
class DesignSpace:
    """Box bounds of the continuous design variables plus the airfoil families."""

    bounds: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "wind_speed": (4.0, 13.0),
            "tsr": (4.0, 12.0),
            "pitch_deg": (-3.0, 8.0),
            "chord_scale": (0.8, 1.2),
            "twist_scale": (0.7, 1.3),
        }
    )
    airfoils: tuple[str, ...] = AIRFOIL_NAMES

    def __post_init__(self) -> None:
        if tuple(self.bounds) != VARIABLES:
            raise ValueError(f"bounds must define exactly {VARIABLES} in this order")
        if any(lo >= hi for lo, hi in self.bounds.values()):
            raise ValueError("every lower bound must be below its upper bound")

    @property
    def lower(self) -> np.ndarray:
        return np.array([b[0] for b in self.bounds.values()])

    @property
    def upper(self) -> np.ndarray:
        return np.array([b[1] for b in self.bounds.values()])

    def scale(self, unit: np.ndarray) -> np.ndarray:
        """Map points from the unit hypercube to the physical bounds."""
        return self.lower + np.asarray(unit) * (self.upper - self.lower)

    def contains(self, raw: np.ndarray) -> np.ndarray:
        """Row-wise check that the continuous variables lie inside the bounds."""
        x = np.atleast_2d(raw)[:, : len(VARIABLES)]
        return np.all((x >= self.lower - 1e-12) & (x <= self.upper + 1e-12), axis=1)


def latin_hypercube(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Latin hypercube sample of ``n`` points in ``[0, 1)^d``.

    Each dimension is cut into ``n`` equal strata, one point is drawn uniformly in
    every stratum and the strata are permuted independently per dimension.
    """
    if n < 1 or d < 1:
        raise ValueError("n and d must be positive")
    jitter = rng.random((n, d))
    strata = np.column_stack([rng.permutation(n) for _ in range(d)])
    return (strata + jitter) / n


def min_pairwise_distance(points: np.ndarray) -> float:
    """Smallest Euclidean distance between two rows (space-filling criterion)."""
    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=2))
    dist[np.diag_indices_from(dist)] = np.inf
    return float(dist.min())


def maximin_latin_hypercube(
    n: int, d: int, rng: np.random.Generator, n_candidates: int = 10
) -> np.ndarray:
    """Best of ``n_candidates`` Latin hypercubes under the maximin distance criterion.

    For large ``n`` the criterion is evaluated on a fixed random subset of 400 rows
    to keep the cost bounded.
    """
    subset = rng.choice(n, size=min(n, 400), replace=False)
    best, best_score = None, -np.inf
    for _ in range(max(1, n_candidates)):
        candidate = latin_hypercube(n, d, rng)
        score = min_pairwise_distance(candidate[subset])
        if score > best_score:
            best, best_score = candidate, score
    assert best is not None
    return best


def sample_design(n: int, space: DesignSpace | None = None, seed: int = 0) -> np.ndarray:
    """Draw a design matrix with columns ``RAW_COLUMNS``.

    The airfoil family is a balanced categorical factor: sample ``i`` gets family
    ``i mod n_families`` after a random shuffle.
    """
    space = space or DesignSpace()
    rng = np.random.default_rng(seed)
    unit = maximin_latin_hypercube(n, len(VARIABLES), rng)
    airfoil = rng.permutation(np.arange(n) % len(space.airfoils)).astype(float)
    return np.column_stack([space.scale(unit), airfoil])
