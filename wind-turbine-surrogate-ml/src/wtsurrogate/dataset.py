"""Dataset generation (BEM runs over a design matrix) and feature engineering."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wtsurrogate.bem import solve_bem
from wtsurrogate.doe import RAW_COLUMNS, DesignSpace, sample_design
from wtsurrogate.rotor import AIR_DENSITY, AIR_VISCOSITY, Rotor, reference_rotor

TARGETS: tuple[str, ...] = ("cp", "ct")
FEATURE_NAMES: tuple[str, ...] = (
    *RAW_COLUMNS,
    "inv_tsr",
    "alpha75_deg",
    "loading75",
    "log10_re75",
)


@dataclass(frozen=True)
class Dataset:
    """Design matrix (columns ``RAW_COLUMNS``) with BEM targets."""

    raw: np.ndarray
    cp: np.ndarray
    ct: np.ndarray
    n_requested: int
    n_failed: int
    bem_seconds: float

    def __len__(self) -> int:
        return self.raw.shape[0]

    def target(self, name: str) -> np.ndarray:
        if name not in TARGETS:
            raise ValueError(f"unknown target '{name}', choose from {TARGETS}")
        return getattr(self, name)

    def to_csv(self, path: str | Path) -> Path:
        """Write the design variables and targets to a CSV file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        table = np.column_stack([self.raw, self.cp, self.ct])
        header = ",".join([*RAW_COLUMNS, *TARGETS])
        np.savetxt(path, table, delimiter=",", header=header, comments="", fmt="%.8g")
        return path


class BemOracle:
    """Evaluate the BEM model on rows of the raw design matrix.

    One reference rotor is built per airfoil family; rows are grouped by family so
    that every group is solved in one vectorised call.
    """

    def __init__(self, space: DesignSpace | None = None, n_elements: int = 30) -> None:
        self.space = space or DesignSpace()
        self.rotors: tuple[Rotor, ...] = tuple(
            reference_rotor(name, n_elements=n_elements) for name in self.space.airfoils
        )
        self.n_calls = 0

    def evaluate(self, raw: np.ndarray) -> dict[str, np.ndarray]:
        """Return ``cp``, ``ct`` and ``converged`` for every row of ``raw``."""
        x = np.atleast_2d(np.asarray(raw, dtype=float))
        out = {"cp": np.empty(len(x)), "ct": np.empty(len(x)),
               "converged": np.zeros(len(x), dtype=bool)}  # fmt: skip
        family = np.rint(x[:, 5]).astype(int)
        for index, rotor in enumerate(self.rotors):
            rows = family == index
            if not rows.any():
                continue
            res = solve_bem(rotor, x[rows, 0], x[rows, 1], np.radians(x[rows, 2]),
                            x[rows, 3], x[rows, 4])  # fmt: skip
            out["cp"][rows], out["ct"][rows] = res.cp, res.ct
            out["converged"][rows] = res.converged
        self.n_calls += len(x)
        return out

    def cp(self, raw: np.ndarray) -> np.ndarray:
        return self.evaluate(raw)["cp"]


def generate_dataset(
    n: int, seed: int = 0, space: DesignSpace | None = None, oracle: BemOracle | None = None
) -> Dataset:
    """Sample a Latin hypercube design and label it with the BEM solver.

    Operating points where any blade element fails to converge are dropped and
    counted in ``n_failed``.
    """
    oracle = oracle or BemOracle(space)
    raw = sample_design(n, oracle.space, seed)
    start = time.perf_counter()
    res = oracle.evaluate(raw)
    elapsed = time.perf_counter() - start
    keep = res["converged"] & np.isfinite(res["cp"]) & np.isfinite(res["ct"])
    return Dataset(raw[keep], res["cp"][keep], res["ct"][keep], n, int((~keep).sum()), elapsed)


class FeatureBuilder:
    """Physics-informed features computed from the raw design variables.

    Added to the six raw columns:

    * ``inv_tsr``: 1 / tip-speed ratio
    * ``alpha75_deg``: angle of attack estimate at 75 percent span, using the ideal
      inflow angle ``atan(2 / (3 * lambda_r))`` (axial induction 1/3, no swirl)
    * ``loading75``: local solidity times local speed ratio at 75 percent span, a
      proxy for the element loading
    * ``log10_re75``: chord Reynolds number at 75 percent span (no induction)

    All features use only the inputs and the known reference geometry, never the
    BEM output, so there is no target leakage.
    """

    names: tuple[str, ...] = FEATURE_NAMES

    def __init__(self, rotors: tuple[Rotor, ...]) -> None:
        self.radius = rotors[0].radius
        self.n_blades = rotors[0].n_blades
        ref = np.array([rotor.at_span(0.75) for rotor in rotors])
        self.chord75, self.twist75 = ref[:, 0], ref[:, 1]

    def transform(self, raw: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(raw, dtype=float))
        u, tsr, pitch, cs, ts = (x[:, i] for i in range(5))
        family = np.rint(x[:, 5]).astype(int)
        lam_r = 0.75 * tsr
        chord = self.chord75[family] * cs
        twist = self.twist75[family] * ts
        phi = np.arctan(2.0 / (3.0 * lam_r))
        alpha = np.degrees(phi - twist) - pitch
        sigma = self.n_blades * chord / (2.0 * np.pi * 0.75 * self.radius)
        reynolds = AIR_DENSITY * u * np.sqrt(1.0 + lam_r**2) * chord / AIR_VISCOSITY
        return np.column_stack([x, 1.0 / tsr, alpha, sigma * lam_r, np.log10(reynolds)])
