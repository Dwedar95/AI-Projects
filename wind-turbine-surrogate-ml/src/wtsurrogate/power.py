"""Power curve and annual energy production (AEP) with a Weibull wind distribution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import gamma

import numpy as np
from scipy.optimize import brentq

from wtsurrogate.bem import solve_bem
from wtsurrogate.rotor import AIR_DENSITY, Rotor

HOURS_PER_YEAR = 8760.0


def weibull_scale(mean_speed: float, shape: float = 2.0) -> float:
    """Weibull scale parameter that gives the requested mean wind speed."""
    return mean_speed / gamma(1.0 + 1.0 / shape)


def weibull_pdf(speed: np.ndarray, scale: float, shape: float = 2.0) -> np.ndarray:
    """Weibull probability density."""
    v = np.asarray(speed, dtype=float) / scale
    return shape / scale * v ** (shape - 1.0) * np.exp(-(v**shape))


def weibull_cdf(speed: np.ndarray | float, scale: float, shape: float = 2.0) -> np.ndarray:
    """Weibull cumulative distribution."""
    return 1.0 - np.exp(-((np.asarray(speed, dtype=float) / scale) ** shape))


@dataclass(frozen=True)
class TurbineLimits:
    """Operating limits used by the simple variable-speed, pitch-regulated controller."""

    rated_power: float = 10.0e6
    drivetrain_efficiency: float = 0.94
    max_tip_speed: float = 90.0
    cut_in: float = 4.0
    cut_out: float = 25.0
    radius: float = 89.0


@dataclass(frozen=True)
class PowerCurve:
    """Electrical power curve with the operating point chosen at every wind speed."""

    speed: np.ndarray
    power: np.ndarray
    cp: np.ndarray
    tsr: np.ndarray
    pitch_deg: np.ndarray

    def rated_speed(self, rated_power: float) -> float:
        """Lowest wind speed at which rated power is reached (NaN if never)."""
        hit = np.flatnonzero(self.power >= rated_power * (1 - 1e-9))
        return float(self.speed[hit[0]]) if hit.size else float("nan")


def compute_power_curve(
    cp_function: Callable[[np.ndarray], np.ndarray],
    airfoil_index: int,
    limits: TurbineLimits | None = None,
    speeds: np.ndarray | None = None,
    max_model_speed: float = 13.0,
    tsr_grid: np.ndarray | None = None,
    pitch_grid: np.ndarray | None = None,
    density: float = AIR_DENSITY,
) -> PowerCurve:
    """Power curve from a Cp model for the baseline geometry (scales equal to 1).

    Below rated power the controller picks the TSR and pitch with the highest Cp on
    a grid, subject to the tip-speed limit. Electrical power is capped at rated
    power. Above ``max_model_speed`` the Cp model is not queried (it is outside the
    sampled design space) and rated power is assumed, which presumes ideal pitch
    regulation in that range.
    """
    limits = limits or TurbineLimits()
    speeds = np.arange(limits.cut_in, limits.cut_out + 1e-9, 0.5) if speeds is None else speeds
    tsr_grid = np.linspace(4.0, 12.0, 33) if tsr_grid is None else tsr_grid
    pitch_grid = np.linspace(-3.0, 8.0, 23) if pitch_grid is None else pitch_grid
    tt, pp = (g.ravel() for g in np.meshgrid(tsr_grid, pitch_grid, indexing="ij"))
    area = np.pi * limits.radius**2
    power = np.full(speeds.size, limits.rated_power)
    cp_out = np.full(speeds.size, np.nan)
    tsr_out = np.full(speeds.size, np.nan)
    pitch_out = np.full(speeds.size, np.nan)
    for i, u in enumerate(speeds):
        if u > max_model_speed:
            continue
        allowed = tt * u <= limits.max_tip_speed + 1e-9
        rows = np.column_stack([np.full(allowed.sum(), u), tt[allowed], pp[allowed],
                                np.ones(allowed.sum()), np.ones(allowed.sum()),
                                np.full(allowed.sum(), float(airfoil_index))])  # fmt: skip
        cp = np.asarray(cp_function(rows), dtype=float)
        best = int(np.nanargmax(cp))
        electrical = 0.5 * density * area * u**3 * cp[best] * limits.drivetrain_efficiency
        power[i] = min(max(electrical, 0.0), limits.rated_power)
        cp_out[i], tsr_out[i], pitch_out[i] = cp[best], rows[best, 1], rows[best, 2]
    return PowerCurve(speeds, power, cp_out, tsr_out, pitch_out)


def annual_energy_production(
    speed: np.ndarray, power: np.ndarray, mean_speed: float, shape: float = 2.0,
    availability: float = 1.0,
) -> float:  # fmt: skip
    """AEP in watt-hours: 8760 h times the Weibull-weighted mean power.

    The power curve is linearly interpolated on a fine grid between its first and
    last wind speed (cut-in and cut-out); power outside that range is zero.
    """
    scale = weibull_scale(mean_speed, shape)
    fine = np.linspace(speed[0], speed[-1], 2001)
    p = np.interp(fine, speed, power)
    mean_power = np.trapezoid(p * weibull_pdf(fine, scale, shape), fine)
    return float(HOURS_PER_YEAR * availability * mean_power)


def capacity_factor(aep_wh: float, rated_power: float) -> float:
    """AEP divided by the energy at continuous rated power."""
    return aep_wh / (rated_power * HOURS_PER_YEAR)


def pitch_for_rated_power(
    rotor: Rotor, wind_speed: float, limits: TurbineLimits | None = None
) -> float:
    """Pitch angle in degrees (towards feather) that holds rated power, from the BEM model.

    The rotor runs at the tip-speed limit. Returns NaN when rated power cannot be
    reached at this wind speed. Used to check the ideal-regulation assumption of the
    power curve above rated wind speed.
    """
    limits = limits or TurbineLimits()
    tsr = limits.max_tip_speed / wind_speed

    def excess(pitch_deg: float) -> float:
        res = solve_bem(rotor, wind_speed, tsr, np.radians(pitch_deg))
        return float(res.power[0]) * limits.drivetrain_efficiency - limits.rated_power

    grid = np.arange(0.0, 35.1, 1.0)
    values = np.array([excess(p) for p in grid])
    crossing = np.flatnonzero((values[:-1] > 0.0) & (values[1:] <= 0.0))
    if crossing.size == 0:
        return float("nan")
    i = int(crossing[-1])
    return float(brentq(excess, grid[i], grid[i + 1], xtol=1e-4))
