"""Composition root: wires broker, ingestion, store, twins and simulators.

:class:`EnergyPlatform` builds one simulated site (PV array, wind turbine,
battery, load) and runs it step by step. Each step publishes one telemetry
message per asset on the broker. Services only see what arrives through the
broker, as they would with field devices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .broker import InProcessBroker
from .dispatch import DispatchConfig, DispatchResult, dispatch_step, summarise
from .ingestion import IngestionService
from .ml.anomaly import ResidualDetector
from .ml.forecast import FeatureConfig, ForecastReport, LagForecaster, evaluate_forecast
from .schemas import Telemetry
from .simulators.devices import DEFAULT_START_TS, Battery, WindTurbine, simulate_weather
from .simulators.pv import PVArray
from .store import TimeSeriesStore
from .twin import Asset, TwinRegistry


@dataclass
class SiteConfig:
    """Configuration of the simulated site."""

    site: str = "demo-site"
    seed: int = 7
    dt_s: float = 900.0
    start_ts: float = DEFAULT_START_TS
    measurement_noise: float = 0.01
    max_days: int = 60
    dispatch: DispatchConfig = field(default_factory=DispatchConfig)


class EnergyPlatform:
    """In-process platform instance with one simulated site."""

    PV_ID = "pv-array-1"
    WIND_ID = "wind-1"
    BATTERY_ID = "battery-1"
    LOAD_ID = "load-1"

    def __init__(self, config: SiteConfig | None = None, db_path: str = ":memory:") -> None:
        self.config = config or SiteConfig()
        self.broker = InProcessBroker()
        self.store = TimeSeriesStore(db_path)
        self.registry = TwinRegistry(stale_after_s=3 * self.config.dt_s)
        self.ingestion = IngestionService(self.broker, self.store, self.registry)
        self.pv = PVArray()
        self.wind = WindTurbine()
        self.battery = Battery()
        self._steps_run = 0
        n_max = int(self.config.max_days * 86400.0 / self.config.dt_s)
        self._weather = simulate_weather(
            n_max, self.config.dt_s, self.config.seed, self.config.start_ts
        )
        self._noise = 1.0 + self.config.measurement_noise * np.random.default_rng(
            self.config.seed + 1
        ).normal(size=n_max)
        self._dispatch_log: list[tuple[float, float, float, float, float, float]] = []
        self._register_assets()

    def _register_assets(self) -> None:
        site = self.config.site
        self.registry.register(Asset(asset_id=site, asset_type="site", site=site, name="Demo site"))
        self.registry.register(
            Asset(
                asset_id=self.PV_ID,
                asset_type="pv",
                site=site,
                parent_id=site,
                name="Rooftop PV array",
                rated_power_w=round(self.pv.rated_power_w, 1),
                properties={
                    "modules_in_series": self.pv.n_series,
                    "strings": self.pv.n_parallel,
                    "model": "single-diode",
                },
            )
        )
        self.registry.register(
            Asset(
                asset_id=self.WIND_ID,
                asset_type="wind",
                site=site,
                parent_id=site,
                name="Small wind turbine",
                rated_power_w=self.wind.rated_power_w,
                properties={"cut_in_ms": self.wind.cut_in_ms, "rated_ms": self.wind.rated_ms},
            )
        )
        self.registry.register(
            Asset(
                asset_id=self.BATTERY_ID,
                asset_type="battery",
                site=site,
                parent_id=site,
                name="Battery storage",
                rated_power_w=self.battery.max_discharge_w,
                properties={"capacity_wh": self.battery.capacity_wh},
            )
        )
        self.registry.register(
            Asset(asset_id=self.LOAD_ID, asset_type="load", site=site, parent_id=site, name="Load")
        )

    def _publish(
        self, asset_id: str, asset_type: str, ts: float, metrics: dict[str, float]
    ) -> None:
        sample = Telemetry(
            site=self.config.site,
            asset_id=asset_id,
            asset_type=asset_type,  # type: ignore[arg-type]
            ts=ts,
            metrics={k: float(v) for k, v in metrics.items()},
            source="simulated",
        )
        self.broker.publish(sample.topic(), sample.model_dump())

    def run(self, hours: float) -> int:
        """Advance the simulation by ``hours`` and return the number of steps run.

        Consecutive calls continue where the previous call stopped. Weather
        and measurement noise are drawn once per platform from the seed, so a
        run is reproducible regardless of how it is split into calls.
        """
        cfg = self.config
        n_new = int(round(hours * 3600.0 / cfg.dt_s))
        total = self._steps_run + n_new
        weather = self._weather
        if total > len(weather.ts):
            raise ValueError(f"simulation is limited to max_days={cfg.max_days}")
        sl = slice(self._steps_run, total)
        pv_point = self.pv.operating_point(weather.irradiance_wm2[sl], weather.ambient_temp_c[sl])
        wind_power = self.wind.power(weather.wind_speed_ms[sl])
        noise = self._noise[sl]
        for j in range(n_new):
            k = self._steps_run + j
            ts = float(weather.ts[k])
            pv_w = float(pv_point["power_w"][j] * noise[j])
            wind_w = float(wind_power[j])
            load_w = float(weather.load_w[k])
            step = dispatch_step(pv_w + wind_w, load_w, self.battery, cfg.dt_s, cfg.dispatch)
            self._publish(
                self.PV_ID,
                "pv",
                ts,
                {
                    "irradiance_wm2": weather.irradiance_wm2[k],
                    "cell_temp_c": pv_point["cell_temp_c"][j],
                    "voltage_v": pv_point["voltage_v"][j],
                    "current_a": pv_point["current_a"][j] * noise[j],
                    "power_w": pv_w,
                },
            )
            self._publish(
                self.WIND_ID,
                "wind",
                ts,
                {"wind_speed_ms": weather.wind_speed_ms[k], "power_w": wind_w},
            )
            self._publish(
                self.BATTERY_ID, "battery", ts, {"soc": self.battery.soc, "power_w": step.battery_w}
            )
            self._publish(self.LOAD_ID, "load", ts, {"power_w": load_w})
            self._dispatch_log.append(
                (
                    pv_w + wind_w,
                    load_w,
                    step.battery_w,
                    step.grid_import_w,
                    step.grid_export_w,
                    self.battery.soc,
                )
            )
        self._steps_run = total
        return n_new

    @property
    def now(self) -> float:
        """Timestamp of the latest simulated step."""
        return self.config.start_ts + self.config.dt_s * max(self._steps_run - 1, 0)

    def dispatch_summary(self) -> DispatchResult:
        """Energy indicators of the dispatch decisions taken so far."""
        if not self._dispatch_log:
            raise ValueError("no simulation steps have been run")
        log = np.asarray(self._dispatch_log)
        return summarise(
            log[:, 0],
            log[:, 1],
            log[:, 2],
            log[:, 3],
            log[:, 4],
            log[:, 5],
            self.config.dt_s,
            self.battery.equivalent_full_cycles,
        )

    def power_series(self, target: str) -> tuple[np.ndarray, np.ndarray]:
        asset_id = {"pv": self.PV_ID, "load": self.LOAD_ID, "wind": self.WIND_ID}.get(target)
        if asset_id is None:
            raise KeyError(f"unknown forecast target {target!r}")
        return self.store.series(asset_id, "power_w")

    def forecast(
        self, target: str, horizon_steps: int = 4
    ) -> tuple[ForecastReport, list[dict[str, float]]]:
        """Backtest and forecast a stored power series.

        Returns the hold-out report for ``horizon_steps`` and one direct
        forecast per lead time from 1 to ``horizon_steps``, each from a model
        trained on the full stored history.
        """
        ts, y = self.power_series(target)
        season = int(round(86400.0 / self.config.dt_s))
        report, _ = evaluate_forecast(
            y,
            ts,
            FeatureConfig(season=season, horizon=horizon_steps),
            target=target,
            seed=self.config.seed,
        )
        points = []
        for h in range(1, horizon_steps + 1):
            model = LagForecaster(FeatureConfig(season=season, horizon=h), self.config.seed)
            target_ts, value = model.fit(y, ts).predict_next(y, ts)
            points.append({"ts": target_ts, "lead_steps": h, "power_w": value})
        return report, points

    def pv_anomalies(self, threshold: float = 4.0) -> list[dict[str, float]]:
        """Flag stored PV samples whose power deviates from the twin model."""
        ts, power = self.store.series(self.PV_ID, "power_w")
        _, g = self.store.series(self.PV_ID, "irradiance_wm2")
        _, t_cell = self.store.series(self.PV_ID, "cell_temp_c")
        if len(ts) == 0:
            return []
        expected = self.pv.expected_power(g, t_cell)
        detector = ResidualDetector(self.pv.rated_power_w, threshold).fit(power, expected)
        scores = detector.score(power, expected)
        return [
            {
                "ts": float(ts[k]),
                "power_w": float(power[k]),
                "expected_w": float(expected[k]),
                "score": float(scores[k]),
            }
            for k in np.flatnonzero(scores > threshold)
        ]
