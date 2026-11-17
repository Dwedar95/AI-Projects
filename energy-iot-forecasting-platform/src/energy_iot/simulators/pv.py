"""Single-diode photovoltaic module model.

The module current is the implicit solution of

    I = Iph - I0 * (exp((V + I*Rs) / a) - 1) - (V + I*Rs) / Rsh

with ``a = n * Ns * k * T / q``. The photocurrent scales with irradiance and
has a small positive temperature coefficient. The saturation current follows
the usual cubic temperature law with the silicon band gap. The equation is
solved with Newton's method, which converges monotonically here because the
residual is concave and decreasing in ``I``.

The parameters describe a generic 60-cell crystalline silicon module. They are
illustrative and are not fitted to any commercial product.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

K_BOLTZMANN = 1.380649e-23  # J/K
Q_ELECTRON = 1.602176634e-19  # C
T_REF_K = 298.15
G_REF = 1000.0
_GOLDEN = (np.sqrt(5.0) - 1.0) / 2.0


@dataclass(frozen=True)
class MPP:
    """Maximum power point of a module at given conditions."""

    voltage_v: float
    current_a: float
    power_w: float


@dataclass(frozen=True)
class PVModule:
    """Single-diode model of one PV module."""

    cells_in_series: int = 60
    iph_ref_a: float = 8.8
    voc_ref_v: float = 37.8
    ideality: float = 1.3
    rs_ohm: float = 0.25
    rsh_ohm: float = 300.0
    alpha_isc_a_per_k: float = 0.0045
    bandgap_ev: float = 1.12
    noct_c: float = 45.0

    def _a(self, t_cell_c: np.ndarray | float) -> np.ndarray:
        """Modified ideality factor n*Ns*Vt in volts."""
        t_k = np.asarray(t_cell_c, dtype=float) + 273.15
        return self.ideality * self.cells_in_series * K_BOLTZMANN * t_k / Q_ELECTRON

    def photocurrent(self, g_wm2: np.ndarray | float, t_cell_c: np.ndarray | float) -> np.ndarray:
        """Light-generated current in amperes."""
        g = np.clip(np.asarray(g_wm2, dtype=float), 0.0, None)
        dt = np.asarray(t_cell_c, dtype=float) - 25.0
        return (self.iph_ref_a + self.alpha_isc_a_per_k * dt) * g / G_REF

    def saturation_current(self, t_cell_c: np.ndarray | float) -> np.ndarray:
        """Diode saturation current in amperes."""
        a_ref = float(self._a(25.0))
        i0_ref = (self.iph_ref_a - self.voc_ref_v / self.rsh_ohm) / np.expm1(self.voc_ref_v / a_ref)
        t_k = np.asarray(t_cell_c, dtype=float) + 273.15
        eg_term = self.bandgap_ev * Q_ELECTRON / (self.ideality * K_BOLTZMANN)
        return i0_ref * (t_k / T_REF_K) ** 3 * np.exp(eg_term * (1.0 / T_REF_K - 1.0 / t_k))

    def current(
        self,
        v: np.ndarray | float,
        g_wm2: np.ndarray | float,
        t_cell_c: np.ndarray | float,
    ) -> np.ndarray:
        """Module current at terminal voltage ``v`` (inputs broadcast)."""
        v, g, t = np.broadcast_arrays(
            np.asarray(v, dtype=float),
            np.asarray(g_wm2, dtype=float),
            np.asarray(t_cell_c, dtype=float),
        )
        iph = self.photocurrent(g, t)
        i0 = self.saturation_current(t)
        a = self._a(t)
        i = iph.copy()
        for _ in range(60):
            arg = np.minimum((v + i * self.rs_ohm) / a, 100.0)
            e = np.exp(arg)
            f = iph - i0 * (e - 1.0) - (v + i * self.rs_ohm) / self.rsh_ohm - i
            df = -i0 * self.rs_ohm / a * e - self.rs_ohm / self.rsh_ohm - 1.0
            step = f / df
            i = i - step
            if np.max(np.abs(step)) < 1e-11:
                break
        return i

    def power(
        self,
        v: np.ndarray | float,
        g_wm2: np.ndarray | float,
        t_cell_c: np.ndarray | float,
    ) -> np.ndarray:
        """Module output power in watts at terminal voltage ``v``."""
        return np.asarray(v, dtype=float) * self.current(v, g_wm2, t_cell_c)

    def open_circuit_voltage(self, g_wm2: float, t_cell_c: float) -> float:
        """Voltage at which the module current is zero (bisection)."""
        if g_wm2 <= 0.0:
            return 0.0
        lo, hi = 0.0, 1.4 * self.voc_ref_v
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if float(self.current(mid, g_wm2, t_cell_c)) > 0.0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def mpp(self, g_wm2: float, t_cell_c: float) -> MPP:
        """True maximum power point, found by bounded scalar optimisation."""
        if g_wm2 <= 0.0:
            return MPP(0.0, 0.0, 0.0)
        voc = self.open_circuit_voltage(g_wm2, t_cell_c)
        res = minimize_scalar(
            lambda v: -float(self.power(v, g_wm2, t_cell_c)),
            bounds=(0.0, voc),
            method="bounded",
            options={"xatol": 1e-7},
        )
        v = float(res.x)
        i = float(self.current(v, g_wm2, t_cell_c))
        return MPP(v, i, v * i)

    def mpp_batch(self, g_wm2: np.ndarray, t_cell_c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised MPP search (golden section). Returns ``(v_mpp, p_mpp)``.

        The power-voltage curve of a uniformly lit single-diode module is
        unimodal, so golden-section search on ``[0, 1.4 * Voc_ref]`` is exact
        up to the bracket tolerance (below 1e-6 V after 45 iterations).
        """
        g = np.atleast_1d(np.asarray(g_wm2, dtype=float))
        t = np.atleast_1d(np.asarray(t_cell_c, dtype=float))
        g, t = np.broadcast_arrays(g, t)
        lo = np.zeros_like(g)
        hi = np.full_like(g, 1.4 * self.voc_ref_v)
        x1 = hi - _GOLDEN * (hi - lo)
        x2 = lo + _GOLDEN * (hi - lo)
        p1 = self.power(x1, g, t)
        p2 = self.power(x2, g, t)
        for _ in range(45):
            left = p1 > p2
            hi = np.where(left, x2, hi)
            lo = np.where(left, lo, x1)
            x1_new = hi - _GOLDEN * (hi - lo)
            x2_new = lo + _GOLDEN * (hi - lo)
            p1 = self.power(x1_new, g, t)
            p2 = self.power(x2_new, g, t)
            x1, x2 = x1_new, x2_new
        v = 0.5 * (lo + hi)
        p = self.power(v, g, t)
        dark = g <= 0.0
        return np.where(dark, 0.0, v), np.where(dark, 0.0, np.maximum(p, 0.0))

    def cell_temperature(
        self, g_wm2: np.ndarray | float, t_ambient_c: np.ndarray | float
    ) -> np.ndarray:
        """NOCT cell temperature estimate from irradiance and ambient temperature."""
        g = np.asarray(g_wm2, dtype=float)
        return np.asarray(t_ambient_c, dtype=float) + (self.noct_c - 20.0) / 800.0 * g


@dataclass(frozen=True)
class PVArray:
    """Array of identical modules behind an ideal MPP tracker and an inverter."""

    module: PVModule = PVModule()
    n_series: int = 6
    n_parallel: int = 2
    inverter_efficiency: float = 0.97

    @property
    def rated_power_w(self) -> float:
        """DC power at standard test conditions (1000 W/m2, 25 C cell)."""
        return self.n_series * self.n_parallel * self.module.mpp(G_REF, 25.0).power_w

    def operating_point(self, g_wm2: np.ndarray, t_ambient_c: np.ndarray) -> dict[str, np.ndarray]:
        """Array voltage, current, AC power and cell temperature at the MPP."""
        t_cell = self.module.cell_temperature(g_wm2, t_ambient_c)
        v, p = self.module.mpp_batch(g_wm2, t_cell)
        v_arr = v * self.n_series
        p_dc = p * self.n_series * self.n_parallel
        i_arr = np.divide(p_dc, v_arr, out=np.zeros_like(p_dc), where=v_arr > 0)
        return {
            "cell_temp_c": np.atleast_1d(t_cell),
            "voltage_v": v_arr,
            "current_a": i_arr,
            "power_w": p_dc * self.inverter_efficiency,
        }

    def expected_power(self, g_wm2: np.ndarray, t_cell_c: np.ndarray) -> np.ndarray:
        """Model-expected AC power from measured irradiance and cell temperature."""
        _, p = self.module.mpp_batch(g_wm2, t_cell_c)
        return p * self.n_series * self.n_parallel * self.inverter_efficiency
