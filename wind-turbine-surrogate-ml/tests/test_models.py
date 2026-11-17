import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from wtsurrogate.dataset import FeatureBuilder
from wtsurrogate.interpret import partial_dependence, permutation_importance
from wtsurrogate.models import (
    MODEL_NAMES,
    Surrogate,
    model_specs,
    regression_metrics,
    tune_and_evaluate,
)
from wtsurrogate.uncertainty import (
    QuantileBoosting,
    conformal_half_width,
    forest_predict_std,
    gaussian_calibration,
    gaussian_interval,
    gp_predict_std,
    interval_quality,
)


@pytest.fixture(scope="module")
def split(dataset, oracle):
    feats = FeatureBuilder(oracle.rotors).transform(dataset.raw)
    return feats[:180], dataset.cp[:180], feats[180:], dataset.cp[180:]


def test_regression_metrics_known_values():
    m = regression_metrics(np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 1.0, 2.0, 5.0]))
    assert m["rmse"] == pytest.approx(1.0)
    assert m["mae"] == pytest.approx(0.5)
    assert m["max_abs_error"] == pytest.approx(2.0)
    assert m["r2"] == pytest.approx(1 - 4 / 5)


def test_model_specs_cover_all_families():
    specs = model_specs(10, budget=0.1)
    assert tuple(specs) == MODEL_NAMES
    assert all(spec.n_iter >= 2 for spec in specs.values())
    assert specs["gaussian_process"].max_train == 500


@pytest.mark.parametrize("name", ["polynomial_ridge", "gaussian_process"])
def test_tuned_surrogates_fit_the_bem_data(name, split):
    x_train, y_train, x_test, y_test = split
    spec = model_specs(x_train.shape[1], budget=0.3)[name]
    report = tune_and_evaluate(spec, x_train, y_train, x_test, y_test, cv=3)
    assert report.test["r2"] > 0.8
    assert report.cv_rmse_mean > 0 and report.search_seconds > 0
    assert "estimator" not in report.summary()
    if name == "gaussian_process":
        mean, std = gp_predict_std(report.estimator, x_test)
        assert np.allclose(mean, report.estimator.predict(x_test))
        assert np.all(std > 0)


def test_surrogate_wrapper_counts_calls(split, dataset, oracle):
    x_train, y_train, _, _ = split
    forest = RandomForestRegressor(n_estimators=20, random_state=0).fit(x_train, y_train)
    surrogate = Surrogate(forest, FeatureBuilder(oracle.rotors))
    pred = surrogate.predict(dataset.raw[:7])
    assert pred.shape == (7,) and surrogate.n_calls == 7
    mean, std = forest_predict_std(forest, x_train[:7])
    assert np.allclose(mean, forest.predict(x_train[:7])) and np.all(std >= 0)


def test_gaussian_interval_and_coverage(rng):
    y = rng.normal(size=20000)
    lo, hi = gaussian_interval(np.zeros_like(y), np.ones_like(y), 0.9)
    assert hi[0] == pytest.approx(1.6449, abs=1e-3)
    assert interval_quality(y, lo, hi)["coverage"] == pytest.approx(0.9, abs=0.01)
    cal = gaussian_calibration(y, np.zeros_like(y), np.ones_like(y))
    assert cal["0.50"] == pytest.approx(0.5, abs=0.015)
    with pytest.raises(ValueError):
        gaussian_interval(y, y, 1.0)


def test_conformal_half_width_gives_coverage(rng):
    calibration, test = rng.standard_t(3, size=500), rng.standard_t(3, size=20000)
    half = conformal_half_width(calibration, 0.9)
    assert half == np.sort(np.abs(calibration))[int(np.ceil(501 * 0.9)) - 1]
    assert np.mean(np.abs(test) <= half) == pytest.approx(0.9, abs=0.03)
    with pytest.raises(ValueError):
        conformal_half_width(np.array([]), 0.9)


def test_quantile_boosting_orders_bounds(split):
    x_train, y_train, x_test, _ = split
    tuned = HistGradientBoostingRegressor(max_iter=60, random_state=0)
    lo, hi = QuantileBoosting(tuned, 0.8).fit(x_train, y_train).interval(x_test)
    assert np.all(lo <= hi) and np.mean(hi - lo) > 0


def test_permutation_importance_and_partial_dependence_on_bem(oracle, dataset):
    """Applied to the BEM model itself: TSR and pitch dominate, wind speed barely matters."""
    raw, y = dataset.raw[:120], dataset.cp[:120]
    imp = permutation_importance(oracle.cp, raw, y, n_repeats=3)
    assert imp["tsr"]["mean"] > imp["wind_speed"]["mean"]
    assert imp["pitch_deg"]["mean"] > 10 * abs(imp["wind_speed"]["mean"])
    grid = np.array([4.0, 7.5, 12.0])
    pdp = partial_dependence(oracle.cp, raw, "tsr", grid)
    assert pdp[1] > pdp[0] and pdp[1] > pdp[2]
