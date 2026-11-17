import numpy as np
import pytest

from anomalybags.ensemble import AnomalyBags, EnsembleConfig, ScoreNormaliser, aggregate_scores
from anomalybags.metrics import roc_auc
from anomalybags.windowing import n_windows


def test_normalisers():
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert ScoreNormaliser("zscore").fit(ref).transform(ref).mean() == pytest.approx(0.0)
    assert ScoreNormaliser("minmax").fit(ref).transform([1.0, 5.0, 9.0]).tolist() == [0.0, 1.0, 2.0]
    assert ScoreNormaliser("rank").fit(ref).transform([0.0, 3.0, 99.0]).tolist() == [0.0, 0.6, 1.0]
    robust = ScoreNormaliser("robust").fit(ref).transform([3.0, 4.0])
    assert robust[0] == 0.0 and robust[1] == pytest.approx(1 / 1.4826)
    assert np.all(np.isfinite(ScoreNormaliser("zscore").fit(np.ones(4)).transform(ref)))
    with pytest.raises(ValueError):
        ScoreNormaliser("softmax")


def test_aggregation_rules():
    m = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [100.0, 1.0]])
    assert aggregate_scores(m, "mean").tolist() == [21.2, 1.0]
    assert aggregate_scores(m, "median").tolist() == [2.0, 1.0]
    assert aggregate_scores(m, "max").tolist() == [100.0, 1.0]
    assert aggregate_scores(m, "trimmed_mean").tolist() == [2.0, 1.0]
    with pytest.raises(ValueError):
        aggregate_scores(m, "product")


def test_config_validation():
    with pytest.raises(ValueError):
        EnsembleConfig(n_estimators=0)
    with pytest.raises(ValueError):
        EnsembleConfig(transforms=("gasf", "fft"))
    with pytest.raises(ValueError):
        EnsembleConfig(sample_frac=1.5)


def test_fit_score_shapes_and_member_plan(small_bench):
    cfg = EnsembleConfig(n_estimators=16, seed=1)
    model = AnomalyBags(cfg).fit(small_bench.train)
    pairs = {(m["transform"], m["detector"]) for m in model.describe()}
    assert len(pairs) == 16  # the full 4 x 4 grid is covered exactly once
    members = model.member_scores(small_bench.test)
    assert members.shape == (16, n_windows(small_bench.test.size, cfg.window, cfg.stride))
    points = model.score_points(small_bench.test)
    assert points.shape == small_bench.test.shape and np.all(np.isfinite(points))
    assert all(m.n_out_of_bag > 0 for m in model.members_)


def test_ensemble_is_deterministic_and_seed_sensitive(small_bench):
    cfg = EnsembleConfig(n_estimators=4, seed=7)
    a = AnomalyBags(cfg).fit(small_bench.train).score_windows(small_bench.test)
    b = AnomalyBags(cfg).fit(small_bench.train).score_windows(small_bench.test)
    c = AnomalyBags(EnsembleConfig(n_estimators=4, seed=8)).fit(small_bench.train)
    assert np.array_equal(a, b)
    assert not np.allclose(a, c.score_windows(small_bench.test))


def test_ensemble_detects_synthetic_anomalies(small_bench):
    model = AnomalyBags(EnsembleConfig(n_estimators=8, seed=0)).fit(small_bench.train)
    assert roc_auc(small_bench.labels, model.score_points(small_bench.test)) > 0.85
    train_scores = model.score_windows(small_bench.train)
    assert np.median(train_scores) < np.median(model.score_windows(small_bench.test)) + 1.0


def test_unfitted_and_short_series_errors():
    with pytest.raises(RuntimeError):
        AnomalyBags().score_windows(np.zeros(500))
    with pytest.raises(ValueError):
        AnomalyBags().fit(np.zeros(100))
