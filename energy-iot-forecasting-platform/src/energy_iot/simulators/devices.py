"""Wind turbine, battery and synthetic weather/load generators.

Everything in this module is a simulation. The weather generator is a simple
stochastic model (clear-sky half sine times an autoregressive cloud factor),
not a reproduction of measured meteorological data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_START_TS = 1_780_272_000.0  # 2026-06-01T00:00:00Z, fixed for reproducibility


@dataclass(frozen=True)
class WindTurbine:
    """Small wind turbine described by a piecewise power curve."""

    rated_power_w: float = 1500.0
    cut_in_ms: float = 3.0
    rated_ms: float = 12.0
    cut_out_ms: float = 25.0

    def power(self, wind_speed_ms: np.ndarray | float) -> np.ndarray:
        """Electrical power: cubic between cut-in and rated speed, flat up to cut-out."""
        v = np.asarray(wind_speed_ms, dtype=float)
        frac = (v**3 - self.cut_in_ms**3) / (self.rated_ms**3 - self.cut_in_ms**3)
        p = self.rated_power_w * np.clip(frac, 0.0, 1.0)
        return np.where((v < self.cut_in_ms) | (v >= self.cut_out_ms), 0.0, p)


@dataclass
class Battery:
    """Energy-bucket battery model with power limits and one-way efficiencies.

    Power sign convention at the terminals: positive charges, negative
    discharges.
    """

    capacity_wh: float = 5000.0
    max_charge_w: float = 2500.0
    max_discharge_w: float = 2500.0
    eta_charge: float = 0.95
    eta_discharge: float = 0.95
    soc: float = 0.5
    soc_min: float = 0.10
    soc_max: float = 0.95
    throughput_wh: float = 0.0

    def step(self, request_w: float, dt_s: float, floor_soc: float | None = None) -> float:
        """Apply a power request for ``dt_s`` seconds and return the power actually used.

        ``floor_soc`` raises the discharge floor above ``soc_min`` for this
        step (used by reserve-keeping dispatch rules).
        """
        dt_h = dt_s / 3600.0
        if request_w >= 0.0:
            headroom_wh = max(self.soc_max - self.soc, 0.0) * self.capacity_wh
            p = min(request_w, self.max_charge_w, headroom_wh / (self.eta_charge * dt_h))
            self.soc += p * self.eta_charge * dt_h / self.capacity_wh
        else:
            floor = self.soc_min if floor_soc is None else max(floor_soc, self.soc_min)
            available_wh = max(self.soc - floor, 0.0) * self.capacity_wh
            p_out = min(-request_w, self.max_discharge_w, available_wh * self.eta_discharge / dt_h)
            self.soc -= p_out / self.eta_discharge * dt_h / self.capacity_wh
            p = -p_out
        self.throughput_wh += abs(p) * dt_h
        return p

    @property
    def equivalent_full_cycles(self) -> float:
        """Terminal energy throughput divided by twice the capacity."""
        return self.throughput_wh / (2.0 * self.capacity_wh)


@dataclass(frozen=True)
class WeatherSeries:
    """Synthetic site conditions on a regular time grid."""

    ts: np.ndarray
    irradiance_wm2: np.ndarray
    ambient_temp_c: np.ndarray
    wind_speed_ms: np.ndarray
    load_w: np.ndarray


def clear_sky_irradiance(
    ts: np.ndarray, sunrise_h: float = 5.5, sunset_h: float = 20.5, peak_wm2: float = 950.0
) -> np.ndarray:
    """Idealised clear-sky plane-of-array irradiance as a function of time of day."""
    hour = (np.asarray(ts, dtype=float) / 3600.0) % 24.0
    x = (hour - sunrise_h) / (sunset_h - sunrise_h)
    shape = np.sin(np.pi * np.clip(x, 0.0, 1.0)) ** 1.2
    return np.where((x > 0.0) & (x < 1.0), peak_wm2 * shape, 0.0)


def _ar1(n: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    """Zero-mean, unit-variance AR(1) sequence."""
    eps = rng.normal(size=n) * np.sqrt(1.0 - phi**2)
    out = np.empty(n)
    prev = rng.normal()
    for k in range(n):
        prev = phi * prev + eps[k]
        out[k] = prev
    return out


def simulate_weather(
    n_steps: int, dt_s: float, seed: int = 0, start_ts: float = DEFAULT_START_TS
) -> WeatherSeries:
    """Generate synthetic irradiance, temperature, wind speed and load.

    Cloud cover is an AR(1) process with a 45 minute correlation time plus a
    per-day offset, so some days are clear and others overcast. The load is a
    household-style profile with a morning and an evening peak.
    """
    rng = np.random.default_rng(seed)
    ts = start_ts + dt_s * np.arange(n_steps)
    hour = (ts / 3600.0) % 24.0
    day = ((ts - start_ts) // 86400.0).astype(int)

    day_offset = rng.normal(0.0, 1.0, size=int(day.max()) + 1)
    z = _ar1(n_steps, float(np.exp(-dt_s / 2700.0)), rng) + day_offset[day]
    cloud = 1.0 / (1.0 + np.exp(-2.0 * (z - 0.6)))  # 0 clear, 1 overcast
    irradiance = clear_sky_irradiance(ts) * (1.0 - 0.8 * cloud)

    temp = (
        18.0
        + 6.0 * np.sin(2.0 * np.pi * (hour - 9.0) / 24.0)
        + 1.5 * _ar1(n_steps, float(np.exp(-dt_s / 14400.0)), rng)
    )
    wind = np.clip(
        6.0
        + 1.0 * np.sin(2.0 * np.pi * (hour - 14.0) / 24.0)
        + 2.8 * _ar1(n_steps, float(np.exp(-dt_s / 7200.0)), rng),
        0.0,
        None,
    )
    weekend = ((day + 0) % 7) >= 5  # start date is a Monday
    load = (
        350.0
        + 550.0 * np.exp(-0.5 * ((hour - 7.5) / 1.2) ** 2)
        + 1100.0 * np.exp(-0.5 * ((hour - 19.0) / 1.8) ** 2)
        + np.where(weekend, 250.0 * np.exp(-0.5 * ((hour - 13.0) / 2.5) ** 2), 0.0)
    )
    load = load * (1.0 + 0.12 * _ar1(n_steps, float(np.exp(-dt_s / 1800.0)), rng))
    return WeatherSeries(ts, irradiance, temp, wind, np.clip(load, 50.0, None))
