import numpy as np
import pytest

from wtsurrogate.dataset import FEATURE_NAMES, FeatureBuilder, generate_dataset
from wtsurrogate.doe import (
    DesignSpace,
    latin_hypercube,
    maximin_latin_hypercube,
    min_pairwise_distance,
    sample_design,
)


def test_latin_hypercube_has_one_point_per_stratum(rng):
    n, d = 40, 5
    sample = latin_hypercube(n, d, rng)
    assert sample.shape == (n, d)
    for col in range(d):
        assert sorted(np.floor(sample[:, col] * n).astype(int)) == list(range(n))


def test_maximin_is_a_latin_hypercube_and_no_worse_than_one_draw():
    best = maximin_latin_hypercube(30, 3, np.random.default_rng(1), n_candidates=20)
    single = maximin_latin_hypercube(30, 3, np.random.default_rng(1), n_candidates=1)
    assert sorted(np.floor(best[:, 0] * 30).astype(int)) == list(range(30))
    assert min_pairwise_distance(best) >= min_pairwise_distance(single)


def test_sample_design_respects_bounds_and_balances_airfoils():
    space = DesignSpace()
    raw = sample_design(101, space, seed=3)
    assert raw.shape == (101, 6)
    assert space.contains(raw).all()
    counts = np.bincount(raw[:, 5].astype(int))
    assert abs(counts[0] - counts[1]) <= 1
    assert np.array_equal(raw, sample_design(101, space, seed=3))


def test_design_space_validation():
    with pytest.raises(ValueError):
        DesignSpace(bounds={"tsr": (4.0, 12.0)})
    bad = dict(DesignSpace().bounds)
    bad["tsr"] = (12.0, 4.0)
    with pytest.raises(ValueError):
        DesignSpace(bounds=bad)


def test_dataset_generation_is_reproducible(oracle, dataset):
    again = generate_dataset(240, seed=1, oracle=oracle)
    assert np.array_equal(dataset.cp, again.cp)
    assert len(dataset) + dataset.n_failed == 240
    assert dataset.cp.max() < 16 / 27 and np.all(np.isfinite(dataset.ct))
    with pytest.raises(ValueError):
        dataset.target("power")


def test_dataset_csv_roundtrip(dataset, tmp_path):
    path = dataset.to_csv(tmp_path / "data.csv")
    header = path.read_text().splitlines()[0]
    assert header == "wind_speed,tsr,pitch_deg,chord_scale,twist_scale,airfoil_index,cp,ct"
    table = np.loadtxt(path, delimiter=",", skiprows=1)
    assert table.shape == (len(dataset), 8)
    assert np.allclose(table[:, 6], dataset.cp, rtol=1e-6, atol=1e-9)


def test_feature_builder_values(oracle):
    builder = FeatureBuilder(oracle.rotors)
    raw = np.array([[9.0, 7.5, 0.0, 1.0, 1.0, 0.0], [9.0, 7.5, 2.0, 1.2, 1.0, 0.0]])
    feats = builder.transform(raw)
    assert feats.shape == (2, len(FEATURE_NAMES))
    names = list(FEATURE_NAMES)
    assert feats[0, names.index("inv_tsr")] == pytest.approx(1 / 7.5)
    alpha = feats[:, names.index("alpha75_deg")]
    assert alpha[0] - alpha[1] == pytest.approx(2.0)  # pitching reduces the angle of attack
    assert 3.0 < alpha[0] < 8.0  # close to the design angle of attack at design TSR
    loading = feats[:, names.index("loading75")]
    assert loading[1] == pytest.approx(1.2 * loading[0])
    assert 6.5 < feats[0, names.index("log10_re75")] < 7.5
