"""End-to-end demo on simulated data. Prints measured metrics from this run.

Run with: python examples/run_demo.py
"""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np

from energy_iot.dispatch import DispatchConfig, simulate_dispatch
from energy_iot.ml.anomaly import (
    IsolationForestDetector,
    ResidualDetector,
    inject_faults,
    score_detection,
)
from energy_iot.ml.forecast import FeatureConfig, evaluate_forecast
from energy_iot.ml.mppt import (
    IncrementalConductance,
    MLSeededPerturbObserve,
    MLVoltagePredictor,
    PerturbObserve,
    compare_trackers,
    irradiance_scenario,
    train_mpp_models,
)
from energy_iot.platform import EnergyPlatform, SiteConfig
from energy_iot.simulators.devices import Battery

SEED = 7


def section(title: str) -> None:
    print(f"\n== {title} ==")


def main() -> None:
    print("NOTE: all data in this demo is SIMULATED (single-diode PV model, synthetic weather).")

    section("1. Event-driven platform: 30 simulated days at 15 minute resolution")
    t0 = time.perf_counter()
    platform = EnergyPlatform(SiteConfig(seed=SEED))
    steps = platform.run(hours=30 * 24)
    stats = platform.ingestion.stats
    print(f"steps simulated          : {steps}")
    print(f"messages published       : {platform.broker.stats.published}")
    print(f"messages accepted        : {stats.accepted}, rejected: {stats.rejected}")
    print(f"points in sqlite store   : {platform.store.count()}")
    print(f"wall time                : {time.perf_counter() - t0:.1f} s")
    bad = {
        "site": "demo-site",
        "asset_id": "pv-array-1",
        "asset_type": "pv",
        "ts": 1.0,
        "metrics": {"power_w": 10.0},
    }
    platform.broker.publish("site/demo-site/pv/pv-array-1/telemetry", bad)
    print(
        f"malformed message        : rejected={platform.ingestion.stats.rejected} "
        f"reasons={platform.ingestion.stats.reasons}"
    )
    twin = platform.registry.describe(platform.PV_ID, platform.now)
    print(
        f"PV twin                  : rated {twin['rated_power_w']} W, "
        f"status {twin['state']['status']}, messages {twin['state']['message_count']}"
    )

    section("2. MPP voltage prediction vs classic MPPT (3000 control periods)")
    module = platform.pv.module
    stc = module.mpp(1000.0, 25.0)
    print(f"module at STC            : Vmpp {stc.voltage_v:.2f} V, Pmpp {stc.power_w:.1f} W")
    models, scores = train_mpp_models(module, seed=SEED)
    for s in scores:
        print(f"regressor {s.name:<20}: hold-out MAE {s.mae_v:.3f} V, R2 {s.r2:.4f}")
    best = min(scores, key=lambda s: s.mae_v).name
    g, t = irradiance_scenario(3000, seed=SEED)
    controllers = [
        PerturbObserve(),
        IncrementalConductance(),
        MLVoltagePredictor(models[best], name=f"ml_voltage_prediction[{best}]"),
        MLSeededPerturbObserve(models[best]),
    ]
    for r in compare_trackers(module, controllers, g, t, seed=SEED):
        print(f"tracking efficiency {r.name:<42}: {100 * r.efficiency:.2f} %")

    aged = replace(module, rs_ohm=2.0 * module.rs_ohm, iph_ref_a=0.92 * module.iph_ref_a)
    print("model mismatch: same controllers on an aged module (Rs x2, Iph -8 %), no retraining")
    for r in compare_trackers(aged, controllers, g, t, seed=SEED):
        print(f"tracking efficiency {r.name:<42}: {100 * r.efficiency:.2f} %")

    section("3. Forecasting 1 hour ahead (4 steps), chronological 75/25 split")
    for target in ("pv", "load"):
        ts, y = platform.power_series(target)
        report, _ = evaluate_forecast(y, ts, FeatureConfig(horizon=4), target=target, seed=SEED)
        print(f"{target:<4} n_train {report.n_train} n_test {report.n_test}")
        print(f"     model       : MAE {report.mae_model:7.1f} W  RMSE {report.rmse_model:7.1f} W")
        print(
            f"     persistence : MAE {report.mae_persistence:7.1f} W  "
            f"RMSE {report.rmse_persistence:7.1f} W"
        )
        print(
            f"     seasonal    : MAE {report.mae_seasonal:7.1f} W  "
            f"RMSE {report.rmse_seasonal:7.1f} W"
        )
        print(
            f"     skill vs persistence {report.skill_vs_persistence:+.3f}, "
            f"vs seasonal {report.skill_vs_seasonal:+.3f}"
        )

    section("4. Anomaly detection on PV telemetry with injected faults")
    ts, power = platform.store.series(platform.PV_ID, "power_w")
    _, irr = platform.store.series(platform.PV_ID, "irradiance_wm2")
    _, t_cell = platform.store.series(platform.PV_ID, "cell_temp_c")
    expected = platform.pv.expected_power(irr, t_cell)
    split = len(power) // 2
    daylight = expected > 0.1 * platform.pv.rated_power_w
    corrupted, labels, events = inject_faults(
        power[split:], daylight[split:], n_faults=25, seed=SEED
    )
    print(
        f"faults injected          : {len(events)} events, {int(labels.sum())} faulty samples "
        f"of {len(labels)}"
    )
    residual = ResidualDetector(platform.pv.rated_power_w).fit(power[:split], expected[:split])
    r = score_detection(labels, residual.predict(corrupted, expected[split:]))
    print(
        f"residual detector        : precision {r.precision:.3f} recall {r.recall:.3f} "
        f"F1 {r.f1:.3f}"
    )
    feats = np.column_stack([irr, t_cell, power])
    forest = IsolationForestDetector(seed=SEED).fit(feats[:split])
    test_feats = np.column_stack([irr[split:], t_cell[split:], corrupted])
    f = score_detection(labels, forest.predict(test_feats))
    print(
        f"isolation forest         : precision {f.precision:.3f} recall {f.recall:.3f} "
        f"F1 {f.f1:.3f}"
    )

    section("5. Rule-based dispatch over the same 30 days (PV + wind, battery 5 kWh)")
    _, wind = platform.store.series(platform.WIND_ID, "power_w")
    _, load = platform.store.series(platform.LOAD_ID, "power_w")
    gen = power + wind
    dt = platform.config.dt_s
    cases = {
        "no battery": simulate_dispatch(gen, load, dt, None),
        "self_consumption": simulate_dispatch(gen, load, dt, Battery()),
        "peak_shaving": simulate_dispatch(
            gen, load, dt, Battery(), DispatchConfig(strategy="peak_shaving")
        ),
    }
    for name, res in cases.items():
        print(
            f"{name:<17}: import {res.grid_import_kwh:6.1f} kWh, export "
            f"{res.grid_export_kwh:6.1f} kWh, peak import {res.peak_import_w:6.0f} W, "
            f"self-sufficiency {100 * res.self_sufficiency:5.1f} %, "
            f"self-consumption {100 * res.self_consumption:5.1f} %, "
            f"cycles {res.battery_cycles:5.1f}"
        )


if __name__ == "__main__":
    main()
