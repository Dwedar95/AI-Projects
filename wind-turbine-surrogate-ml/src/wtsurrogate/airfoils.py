"""Approximate analytical lift and drag polars for two NACA 4-digit airfoils.

These polars are APPROXIMATIONS built from textbook models, not measured or CFD
data:

* attached flow: thin-airfoil theory. The zero-lift angle is integrated from the
  NACA 4-digit camber line, the lift slope is ``2 * pi`` times a constant viscous
  correction factor
* stall: a logistic blend between the attached polar and a flat-plate model
  (``cl = cd_max / 2 * sin(2 alpha)``, ``cd = cd_max * sin(alpha)^2``), in the spirit
  of the Viterna-Corrigan extrapolation
* drag: a parabolic drag bucket around the lift at zero angle of attack, with a
  weak power-law Reynolds number dependence of the minimum drag

Stall angles and minimum drag values are rounded, assumed values of the right order
of magnitude for 15 percent thick sections. They are not fitted to the polars used
in the author's paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def naca4_camber_slope(x: np.ndarray, m: float, p: float) -> np.ndarray:
    """Slope ``dz/dx`` of the NACA 4-digit mean camber line at chordwise position ``x``."""
    x = np.asarray(x, dtype=float)
    if m == 0.0:
        return np.zeros_like(x)
    front = 2.0 * m / p**2 * (p - x)
    rear = 2.0 * m / (1.0 - p) ** 2 * (p - x)
    return np.where(x < p, front, rear)


def thin_airfoil_zero_lift_angle(m: float, p: float, n: int = 4001) -> float:
    """Zero-lift angle of attack in radians from thin-airfoil theory.

    ``alpha_L0 = -(1 / pi) * integral_0^pi dz/dx * (cos(theta) - 1) d(theta)`` with
    ``x = (1 - cos(theta)) / 2``.
    """
    theta = np.linspace(0.0, np.pi, n)
    slope = naca4_camber_slope(0.5 * (1.0 - np.cos(theta)), m, p)
    return float(-np.trapezoid(slope * (np.cos(theta) - 1.0), theta) / np.pi)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


@dataclass(frozen=True)
class AirfoilPolar:
    """Analytical polar of one airfoil family. Angles are stored in radians."""

    name: str
    camber: float
    camber_position: float
    thickness: float
    stall_angle: float
    negative_stall_angle: float
    cd_min_ref: float
    slope_factor: float = 0.9
    stall_width: float = np.radians(2.0)
    drag_bucket: float = 0.008
    cd_max: float = 1.4
    reynolds_ref: float = 6.0e6
    reynolds_exponent: float = 0.15

    @property
    def zero_lift_angle(self) -> float:
        return thin_airfoil_zero_lift_angle(self.camber, self.camber_position)

    @property
    def lift_slope(self) -> float:
        """Lift curve slope per radian in attached flow."""
        return 2.0 * np.pi * self.slope_factor

    def attachment(self, alpha: np.ndarray) -> np.ndarray:
        """Blend weight in [0, 1]: 1 for attached flow, 0 for fully separated flow."""
        upper = _sigmoid((self.stall_angle - alpha) / self.stall_width)
        lower = _sigmoid((alpha - self.negative_stall_angle) / self.stall_width)
        return upper * lower

    def coefficients(
        self, alpha: np.ndarray, reynolds: np.ndarray | float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Lift and drag coefficient at angle of attack ``alpha`` (radians)."""
        alpha = np.asarray(alpha, dtype=float)
        re = self.reynolds_ref if reynolds is None else reynolds
        re = np.clip(np.asarray(re, dtype=float), 1.0e5, 5.0e7)
        cd_min = self.cd_min_ref * (self.reynolds_ref / re) ** self.reynolds_exponent
        cl_att = self.lift_slope * (alpha - self.zero_lift_angle)
        cl_bucket = -self.lift_slope * self.zero_lift_angle
        cd_att = cd_min + self.drag_bucket * (cl_att - cl_bucket) ** 2
        cl_sep = 0.5 * self.cd_max * np.sin(2.0 * alpha)
        cd_sep = cd_min + self.cd_max * np.sin(alpha) ** 2
        f = self.attachment(alpha)
        cl = f * cl_att + (1.0 - f) * cl_sep
        cd = f * np.minimum(cd_att, cd_sep + 0.5) + (1.0 - f) * cd_sep
        return cl, cd

    def best_glide(self) -> tuple[float, float, float]:
        """Angle of attack (rad) with maximum ``cl / cd`` at the reference Reynolds number.

        Returns ``(alpha, cl, cl / cd)``.
        """
        alpha = np.radians(np.linspace(-2.0, 14.0, 1601))
        cl, cd = self.coefficients(alpha)
        best = int(np.argmax(cl / cd))
        return float(alpha[best]), float(cl[best]), float(cl[best] / cd[best])


AIRFOILS: dict[str, AirfoilPolar] = {
    "naca4415": AirfoilPolar(
        name="naca4415", camber=0.04, camber_position=0.4, thickness=0.15,
        stall_angle=np.radians(14.0), negative_stall_angle=np.radians(-12.0), cd_min_ref=0.0068,
    ),
    "naca2215": AirfoilPolar(
        name="naca2215", camber=0.02, camber_position=0.2, thickness=0.15,
        stall_angle=np.radians(13.0), negative_stall_angle=np.radians(-12.0), cd_min_ref=0.0062,
    ),
}  # fmt: skip

AIRFOIL_NAMES: tuple[str, ...] = tuple(AIRFOILS)


def get_airfoil(name: str) -> AirfoilPolar:
    """Look up an airfoil polar by name."""
    try:
        return AIRFOILS[name]
    except KeyError as err:
        raise ValueError(f"unknown airfoil '{name}', choose from {AIRFOIL_NAMES}") from err
