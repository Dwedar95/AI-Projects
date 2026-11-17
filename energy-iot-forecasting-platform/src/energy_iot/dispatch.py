"""Rule-based dispatch for a PV + battery + load site with a grid connection.

Two strategies:

* ``self_consumption``: surplus PV charges the battery, any deficit is
  served from the battery first, the grid covers the rest.
* ``peak_shaving``: like self consumption, but the battery keeps a reserve.
  Below the reserve it only discharges to hold the grid import under a limit.

The rules are deliberately simple and transparent. This is not an optimal
(for example MILP or MPC) scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .simulators.devices import Battery

Strategy = Literal["self_consumption", "peak_shaving"]


@dataclass(frozen=True)
class DispatchConfig:
    """Parameters of the dispatch rules."""

    strategy: Strategy = "self_consumption"
    grid_import_limit_w: float = 1000.0
    reserve_soc: float = 0.35


@dataclass(frozen=True)
class DispatchStep:
    """Power flows of one step in watts (all non-negative except ``battery_w``)."""

    battery_w: float  # positive charging, negative discharging
    grid_import_w: float
    grid_export_w: float


def dispatch_step(
    generation_w: float, load_w: float, battery: Battery, dt_s: float, cfg: DispatchConfig
) -> DispatchStep:
    """Decide the battery power for one step and return the resulting grid exchange."""
    net = generation_w - load_w
    if net >= 0.0:
        charge = battery.step(net, dt_s)
        return DispatchStep(charge, 0.0, net - charge)
    deficit = -net
    if cfg.strategy == "self_consumption":
        discharge = -battery.step(-deficit, dt_s)
    else:
        # Above the reserve: cover the deficit. Below it: only shave the peak.
        discharge = -battery.step(-deficit, dt_s, floor_soc=cfg.reserve_soc)
        over_limit = deficit - discharge - cfg.grid_import_limit_w
        if over_limit > 1e-9:
            headroom = max(battery.max_discharge_w - discharge, 0.0)
            discharge += -battery.step(-min(over_limit, headroom), dt_s)
    return DispatchStep(-discharge, deficit - discharge, 0.0)


@dataclass(frozen=True)
class DispatchResult:
    """Time series and summary indicators of a dispatch run."""

    battery_w: np.ndarray
    grid_import_w: np.ndarray
    grid_export_w: np.ndarray
    soc: np.ndarray
    generation_kwh: float
    load_kwh: float
    grid_import_kwh: float
    grid_export_kwh: float
    peak_import_w: float
    self_consumption: float
    self_sufficiency: float
    battery_cycles: float

    def summary(self) -> dict[str, float]:
        """Scalar indicators only."""
        return {
            "generation_kwh": self.generation_kwh,
            "load_kwh": self.load_kwh,
            "grid_import_kwh": self.grid_import_kwh,
            "grid_export_kwh": self.grid_export_kwh,
            "peak_import_w": self.peak_import_w,
            "self_consumption": self.self_consumption,
            "self_sufficiency": self.self_sufficiency,
            "battery_cycles": self.battery_cycles,
        }


def summarise(
    generation_w: np.ndarray,
    load_w: np.ndarray,
    battery_w: np.ndarray,
    grid_import_w: np.ndarray,
    grid_export_w: np.ndarray,
    soc: np.ndarray,
    dt_s: float,
    battery_cycles: float,
) -> DispatchResult:
    """Compute energy indicators from power series.

    Self consumption is the share of generation not exported. Self sufficiency
    is the share of load not imported.
    """
    k = dt_s / 3.6e6  # W over dt_s to kWh
    gen_kwh = float(np.sum(generation_w) * k)
    load_kwh = float(np.sum(load_w) * k)
    imp_kwh = float(np.sum(grid_import_w) * k)
    exp_kwh = float(np.sum(grid_export_w) * k)
    return DispatchResult(
        battery_w=np.asarray(battery_w),
        grid_import_w=np.asarray(grid_import_w),
        grid_export_w=np.asarray(grid_export_w),
        soc=np.asarray(soc),
        generation_kwh=gen_kwh,
        load_kwh=load_kwh,
        grid_import_kwh=imp_kwh,
        grid_export_kwh=exp_kwh,
        peak_import_w=float(np.max(grid_import_w)) if len(grid_import_w) else 0.0,
        self_consumption=1.0 - exp_kwh / gen_kwh if gen_kwh > 0 else 0.0,
        self_sufficiency=1.0 - imp_kwh / load_kwh if load_kwh > 0 else 0.0,
        battery_cycles=battery_cycles,
    )


def simulate_dispatch(
    generation_w: np.ndarray,
    load_w: np.ndarray,
    dt_s: float,
    battery: Battery | None,
    cfg: DispatchConfig | None = None,
) -> DispatchResult:
    """Run the dispatch rules over whole series. ``battery=None`` gives the no-storage case."""
    cfg = cfg or DispatchConfig()
    n = len(generation_w)
    batt = (
        battery
        if battery is not None
        else Battery(capacity_wh=1.0, max_charge_w=0.0, max_discharge_w=0.0)
    )
    b, imp, exp, soc = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    for k in range(n):
        step = dispatch_step(float(generation_w[k]), float(load_w[k]), batt, dt_s, cfg)
        b[k], imp[k], exp[k], soc[k] = (
            step.battery_w,
            step.grid_import_w,
            step.grid_export_w,
            batt.soc,
        )
    cycles = batt.equivalent_full_cycles if battery is not None else 0.0
    return summarise(generation_w, load_w, b, imp, exp, soc, dt_s, cycles)
