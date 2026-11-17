"""Zonotopes: the set representation used for reachability analysis.

A zonotope is Z = {c + G b : b in [-1, 1]^m} with centre c (n,) and generator matrix
G (n, m). Zonotopes are closed under linear maps and Minkowski sums, which is exactly
what propagating a linear(ised) system with bounded disturbances needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class Zonotope:
    """Centre/generator zonotope in R^n."""

    center: np.ndarray
    generators: np.ndarray

    def __post_init__(self) -> None:
        c = np.asarray(self.center, dtype=np.float64).reshape(-1)
        g = np.asarray(self.generators, dtype=np.float64)
        if g.size == 0:
            g = np.zeros((c.size, 0))
        if g.ndim != 2 or g.shape[0] != c.size:
            raise ValueError("generator matrix must have shape (n, m) matching the centre")
        object.__setattr__(self, "center", c)
        object.__setattr__(self, "generators", g)

    # ------------------------------------------------------------------ constructors
    @staticmethod
    def from_box(lower: np.ndarray, upper: np.ndarray) -> Zonotope:
        """Axis-aligned box [lower, upper] as a zonotope."""
        lo = np.asarray(lower, dtype=np.float64).reshape(-1)
        hi = np.asarray(upper, dtype=np.float64).reshape(-1)
        if np.any(hi < lo):
            raise ValueError("upper bound below lower bound")
        return Zonotope(0.5 * (lo + hi), np.diag(0.5 * (hi - lo)))

    @staticmethod
    def point(x: np.ndarray) -> Zonotope:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        return Zonotope(x, np.zeros((x.size, 0)))

    # ------------------------------------------------------------------ properties
    @property
    def dim(self) -> int:
        return self.center.size

    @property
    def num_generators(self) -> int:
        return self.generators.shape[1]

    @property
    def order(self) -> float:
        """Number of generators per dimension."""
        return self.num_generators / self.dim

    # ------------------------------------------------------------------ set operations
    def linear_map(self, matrix: np.ndarray) -> Zonotope:
        """Image of the set under x -> M x."""
        m = np.asarray(matrix, dtype=np.float64)
        return Zonotope(m @ self.center, m @ self.generators)

    def translate(self, offset: np.ndarray) -> Zonotope:
        return Zonotope(self.center + np.asarray(offset, dtype=np.float64), self.generators)

    def minkowski_sum(self, other: Zonotope) -> Zonotope:
        """{a + b : a in self, b in other}."""
        if other.dim != self.dim:
            raise ValueError("dimension mismatch in Minkowski sum")
        return Zonotope(self.center + other.center, np.hstack([self.generators, other.generators]))

    def __add__(self, other: Zonotope) -> Zonotope:
        return self.minkowski_sum(other)

    def project(self, dims: list[int] | tuple[int, ...]) -> Zonotope:
        """Projection onto a subset of coordinates."""
        idx = list(dims)
        return Zonotope(self.center[idx], self.generators[idx, :])

    def cartesian_product(self, other: Zonotope) -> Zonotope:
        """Zonotope of stacked vectors (x, y) with x in self and y in other."""
        top = np.hstack([self.generators, np.zeros((self.dim, other.num_generators))])
        bottom = np.hstack([np.zeros((other.dim, self.num_generators)), other.generators])
        return Zonotope(np.concatenate([self.center, other.center]), np.vstack([top, bottom]))

    # ------------------------------------------------------------------ enclosures
    def radius(self) -> np.ndarray:
        """Half-widths of the interval hull."""
        return np.abs(self.generators).sum(axis=1)

    def interval_hull(self) -> tuple[np.ndarray, np.ndarray]:
        """Tightest axis-aligned box containing the set, as (lower, upper)."""
        r = self.radius()
        return self.center - r, self.center + r

    def support(self, direction: np.ndarray) -> float:
        """Support function max {d . x : x in Z}."""
        d = np.asarray(direction, dtype=np.float64)
        return float(d @ self.center + np.abs(d @ self.generators).sum())

    def reduce_order(self, max_order: float) -> Zonotope:
        """Girard's order reduction: an over-approximation with at most max_order * n generators.

        The generators that are least box-like (smallest 1-norm minus infinity-norm) are
        replaced by one axis-aligned box that encloses their sum.
        """
        n, m = self.generators.shape
        limit = int(np.floor(max_order * n))
        if m <= limit or m <= n:
            return self
        keep = max(limit - n, 0)
        g = self.generators
        score = np.abs(g).sum(axis=0) - np.abs(g).max(axis=0)
        order = np.argsort(score)
        boxed, kept = order[: m - keep], order[m - keep:]
        box = np.diag(np.abs(g[:, boxed]).sum(axis=1))
        return Zonotope(self.center, np.hstack([g[:, kept], box]))

    # ------------------------------------------------------------------ queries
    def contains_point(self, point: np.ndarray, tol: float = 1e-9) -> bool:
        """Exact membership test by linear programming.

        Solves min t subject to c + G b = p, |b_i| <= t. The point is inside iff t <= 1.
        """
        p = np.asarray(point, dtype=np.float64).reshape(-1)
        lo, hi = self.interval_hull()
        if np.any(p < lo - tol) or np.any(p > hi + tol):
            return False
        m = self.num_generators
        if m == 0:
            return bool(np.allclose(p, self.center, atol=tol))
        return _min_scale(self.generators, p - self.center) <= 1.0 + tol

    def intersects(self, other: Zonotope, tol: float = 1e-9) -> bool:
        """Exact zonotope-vs-zonotope intersection test.

        Z1 and Z2 intersect iff c2 - c1 lies in the zonotope with generators [G1, G2].
        A cheap interval-hull test rules out most pairs before the LP is solved.
        """
        if other.dim != self.dim:
            raise ValueError("dimension mismatch in intersection test")
        lo1, hi1 = self.interval_hull()
        lo2, hi2 = other.interval_hull()
        if np.any(hi1 < lo2 - tol) or np.any(hi2 < lo1 - tol):
            return False
        g = np.hstack([self.generators, other.generators])
        if g.shape[1] == 0:
            return bool(np.allclose(self.center, other.center, atol=tol))
        return _min_scale(g, other.center - self.center) <= 1.0 + tol

    def intersects_box(self, lower: np.ndarray, upper: np.ndarray, tol: float = 1e-9) -> bool:
        """Exact zonotope-vs-box intersection test."""
        return self.intersects(Zonotope.from_box(lower, upper), tol)

    def inside_box(self, lower: np.ndarray, upper: np.ndarray, tol: float = 1e-9) -> bool:
        """True iff the whole set lies inside the box. Exact, via the interval hull."""
        lo, hi = self.interval_hull()
        return bool(np.all(lo >= np.asarray(lower) - tol) and np.all(hi <= np.asarray(upper) + tol))

    def sample(self, rng: np.random.Generator, n: int = 1, boundary: bool = False) -> np.ndarray:
        """Random points of the set, shape (n, dim). With ``boundary`` the generator
        coefficients are +-1, which gives extreme points that stress containment tests."""
        m = self.num_generators
        if boundary:
            b = rng.choice([-1.0, 1.0], size=(n, m))
        else:
            b = rng.uniform(-1.0, 1.0, size=(n, m))
        return self.center + b @ self.generators.T

    def vertices_2d(self) -> np.ndarray:
        """Vertices of a 2D zonotope in counter-clockwise order (for plotting)."""
        if self.dim != 2:
            raise ValueError("vertices_2d needs a 2D zonotope")
        g = self.generators[:, np.abs(self.generators).sum(axis=0) > 1e-12]
        if g.shape[1] == 0:
            return self.center[None, :].copy()
        g = np.where(g[1] < 0, -g, g)  # point every generator into the upper half plane
        g = g[:, np.argsort(np.arctan2(g[1], g[0]))]
        upper = -g.sum(axis=1) + 2.0 * np.cumsum(g, axis=1).T
        start = -g.sum(axis=1)
        half = np.vstack([start, upper])
        return self.center + np.vstack([half[:-1], -half[:-1]])


def convex_hull_enclosure(z1: Zonotope, z2: Zonotope) -> Zonotope:
    """Zonotope that contains the convex hull of two zonotopes (Girard, HSCC 2005).

    Used to cover the straight segment between two consecutive reachable sets. The
    generator matrices are zero-padded to the same width; the enclosure is valid for
    any pairing of generators.
    """
    if z1.dim != z2.dim:
        raise ValueError("dimension mismatch in convex hull enclosure")
    m = max(z1.num_generators, z2.num_generators)
    g1 = np.hstack([z1.generators, np.zeros((z1.dim, m - z1.num_generators))])
    g2 = np.hstack([z2.generators, np.zeros((z2.dim, m - z2.num_generators))])
    half_diff = 0.5 * (z1.center - z2.center)
    return Zonotope(0.5 * (z1.center + z2.center),
                    np.hstack([0.5 * (g1 + g2), half_diff[:, None], 0.5 * (g1 - g2)]))


def _min_scale(generators: np.ndarray, target: np.ndarray) -> float:
    """Smallest t with target = G b and |b|_inf <= t (inf if target is not in span G)."""
    n, m = generators.shape
    cost = np.zeros(m + 1)
    cost[-1] = 1.0
    a_eq = np.hstack([generators, np.zeros((n, 1))])
    eye = np.eye(m)
    ones = np.ones((m, 1))
    a_ub = np.vstack([np.hstack([eye, -ones]), np.hstack([-eye, -ones])])
    res = linprog(cost, A_ub=a_ub, b_ub=np.zeros(2 * m), A_eq=a_eq, b_eq=target,
                  bounds=[(None, None)] * m + [(0.0, None)], method="highs")
    return float(res.x[-1]) if res.status == 0 else float("inf")
