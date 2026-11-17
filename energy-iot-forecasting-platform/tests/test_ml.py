import numpy as np
import pytest

from energy_iot.dispatch import DispatchConfig, simulate_dispatch
from energy_iot.ml.anomaly import (
    IsolationForestDetector,
    ResidualDetector,
    inject_faults,
    score_detection,
)
from energy_iot.ml.forecast import (
    FeatureConfig,
    LagForecaster,
    evaluate_forecast,
    make_supervised,
    skill_score,
)
from energy_iot.ml.mppt import (
    IncrementalConductance,
    MLSeededPerturbObserve,
    MLVoltagePredictor,
    PerturbObserve,
    SensorNoise,
    irradiance_scenario,
    simulate_tracking,
    train_mpp_models,
)
from energy_iot.simulators import Battery, PVModule

MODULE = PVModule()


@pytest.fixture(scope="module")
def trained():
    return train_mpp_models(MODULE, n_train=800, n_test=300, seed=1)


def test_classic_trackers_converge_under_constant_conditions():
    g, t = np.full(400, 800.0), np.full(400, 35.0)
    quiet = SensorNoise(0.0, 0.0, 0.0)
    v_mpp = MODULE.mpp(800.0, 35.0).voltage_v
    for controller in (PerturbObserve(0.2), IncrementalConductance(0.2)):
        res = simulate_tracking(MODULE, controller, g, t, noise=quiet, v_init=20.0)
        assert abs(res.voltage_v[-50:].mean() - v_mpp) < 0.5
        assert res.power_w[-50:].sum() / res.mpp_power_w[-50:].sum() > 0.995


def test_tracking_efficiency_is_bounded_by_one():
    g, t = irradiance_scenario(600, seed=2)
    res = simulate_tracking(MODULE, PerturbObserve(), g, t, seed=2)
    assert 0.9 < res.efficiency <= 1.0
    assert np.all(res.power_w <= res.mpp_power_w + 1e-6)


def test_mpp_regressors_are_accurate(trained):
    _, scores = trained
    assert {s.name for s in scores} == {
        "linear_log_features",
        "random_forest",
        "gradient_boosting",
    }
    for s in scores:
        assert s.mae_v < 0.3 and s.r2 > 0.98


def test_ml_tracker_beats_perturb_observe_on_matched_module(trained):
    models, _ = trained
    g, t = irradiance_scenario(900, seed=4)
    po = simulate_tracking(MODULE, PerturbObserve(), g, t, seed=4)
    ml = simulate_tracking(MODULE, MLVoltagePredictor(models["gradient_boosting"]), g, t, seed=4)
    hybrid = simulate_tracking(
        MODULE, MLSeededPerturbObserve(models["gradient_boosting"]), g, t, seed=4
    )
    assert ml.efficiency > po.efficiency
    assert hybrid.efficiency > 0.99


def _daily_series(days=12, seed=0):
    rng = np.random.default_rng(seed)
    ts = 900.0 * np.arange(96 * days)
    hour = (ts / 3600.0) % 24.0
    y = 500.0 + 300.0 * np.sin(2 * np.pi * hour / 24.0) + rng.normal(0.0, 20.0, len(ts))
    return y, ts


def test_supervised_rows_use_only_past_values():
    y, ts = _daily_series(3)
    cfg = FeatureConfig(n_lags=4, season=96, horizon=4)
    x, target, origins = make_supervised(y, ts, cfg)
    k = 10
    t0 = origins[k]
    assert target[k] == y[t0 + 4]
    assert np.array_equal(x[k, :4], y[[t0, t0 - 1, t0 - 2, t0 - 3]])
    assert x[k, 4] == y[t0 + 4 - 96]
    # Changing the future must not change the features of this origin.
    y2 = y.copy()
    y2[t0 + 1 :] += 1000.0
    x2, _, _ = make_supervised(y2, ts, cfg)
    assert np.array_equal(x[k], x2[k])


def test_feature_config_validation():
    with pytest.raises(ValueError):
        FeatureConfig(horizon=200, season=96)
    with pytest.raises(ValueError):
        make_supervised(np.zeros(20), np.arange(20.0), FeatureConfig())


def test_forecaster_beats_persistence_on_seasonal_series():
    y, ts = _daily_series()
    report, model = evaluate_forecast(y, ts, FeatureConfig(horizon=8), target="synthetic")
    assert report.n_train + report.n_test == len(y) - 8 - (96 - 8)
    assert report.rmse_model < report.rmse_persistence
    assert report.skill_vs_persistence > 0.5
    target_ts, value = model.predict_next(y, ts)
    assert target_ts == ts[-1] + 8 * 900.0 and value >= 0.0


def test_skill_score_definition():
    assert skill_score(50.0, 100.0) == pytest.approx(0.5)
    assert skill_score(100.0, 100.0) == pytest.approx(0.0)
    assert skill_score(120.0, 100.0) < 0
    with pytest.raises(RuntimeError):
        LagForecaster().predict_origins(np.zeros(200), np.arange(200.0), np.array([150]))


def test_fault_injection_labels_and_residual_detector():
    rng = np.random.default_rng(0)
    expected = np.tile(np.concatenate([np.zeros(48), 1500 * np.sin(np.linspace(0, np.pi, 48))]), 20)
    measured = expected * (1.0 + 0.01 * rng.normal(size=len(expected)))
    eligible = expected > 300.0
    corrupted, labels, events = inject_faults(measured, eligible, n_faults=15, seed=1)
    assert len(events) > 5 and labels.sum() > 0
    assert np.array_equal(corrupted[~labels], measured[~labels])
    assert np.all(eligible[[start for start, _ in events]])
    detector = ResidualDetector(rated_power_w=3000.0).fit(measured, expected)
    assert detector.predict(measured, expected).mean() < 0.02  # few false alarms when healthy
    score = score_detection(labels, detector.predict(corrupted, expected))
    assert score.recall > 0.8 and score.precision > 0.7
    # Night samples are never flagged.
    assert not detector.predict(corrupted, expected)[expected == 0.0].any()


def test_isolation_forest_flags_gross_outliers_and_scoring():
    rng = np.random.default_rng(0)
    g = rng.uniform(100, 1000, 500)
    feats = np.column_stack([g, 20 + 0.03 * g, 3.0 * g])
    detector = IsolationForestDetector(contamination=0.02, seed=0).fit(feats)
    # Values outside the marginal ranges are isolated quickly. Points that only break
    # the correlation between features are a known weak spot of axis-parallel splits,
    # which is why the residual detector exists.
    outliers = np.array([[900.0, 47.0, 0.0], [1000.0, 50.0, 6000.0]])
    assert detector.predict(outliers).all()
    assert detector.predict(feats).mean() < 0.05
    s = score_detection(np.array([1, 1, 0, 0], bool), np.array([1, 0, 1, 0], bool))
    assert (s.precision, s.recall, s.f1) == (0.5, 0.5, 0.5)


def test_dispatch_battery_reduces_import_and_conserves_energy():
    n = 96 * 2
    hour = (np.arange(n) * 0.25) % 24
    gen = 3000.0 * np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None)
    load = np.full(n, 800.0)
    base = simulate_dispatch(gen, load, 900.0, None)
    batt = simulate_dispatch(gen, load, 900.0, Battery())
    assert base.battery_cycles == 0.0 and np.all(base.battery_w == 0.0)
    assert batt.grid_import_kwh < base.grid_import_kwh
    assert batt.self_sufficiency > base.self_sufficiency
    balance = gen + batt.grid_import_w - load - batt.battery_w - batt.grid_export_w
    assert np.max(np.abs(balance)) < 1e-6
    assert batt.soc.min() >= 0.10 - 1e-9 and batt.soc.max() <= 0.95 + 1e-9
    assert np.all((batt.grid_import_w == 0) | (batt.grid_export_w == 0))


def test_peak_shaving_keeps_reserve_for_peaks():
    load = np.concatenate([np.full(40, 600.0), np.full(4, 2000.0)])
    gen = np.zeros_like(load)
    cfg = DispatchConfig(strategy="peak_shaving", grid_import_limit_w=1000.0, reserve_soc=0.4)
    shaving = simulate_dispatch(gen, load, 900.0, Battery(soc=0.5), cfg)
    greedy = simulate_dispatch(gen, load, 900.0, Battery(soc=0.5))
    assert shaving.grid_import_w[-4:].max() == pytest.approx(1000.0)
    assert greedy.grid_import_w[-4:].max() == pytest.approx(2000.0)
    assert shaving.peak_import_w < greedy.peak_import_w
