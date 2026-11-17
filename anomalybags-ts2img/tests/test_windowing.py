import numpy as np
import pytest

from anomalybags.windowing import n_windows, sliding_windows, window_labels, windows_to_points


def test_window_count_and_content():
    x = np.arange(20.0)
    w = sliding_windows(x, length=8, stride=4)
    assert w.shape == (n_windows(20, 8, 4), 8) == (4, 8)
    assert np.array_equal(w[1], np.arange(4.0, 12.0))


def test_too_short_series_raises():
    with pytest.raises(ValueError):
        sliding_windows(np.arange(5.0), length=8)


def test_window_labels_any_rule():
    y = np.zeros(20, dtype=int)
    y[9] = 1
    assert window_labels(y, 8, 4).tolist() == [0, 1, 1, 0]


def test_points_from_constant_window_scores():
    scores = np.full(n_windows(50, 10, 5), 2.5)
    assert np.allclose(windows_to_points(scores, 50, 10, 5), 2.5)


def test_points_mean_matches_brute_force(rng):
    n, length, stride = 37, 8, 3
    scores = rng.normal(size=n_windows(n, length, stride))
    expected = np.zeros(n)
    count = np.zeros(n)
    for i, s in enumerate(scores):
        expected[i * stride : i * stride + length] += s
        count[i * stride : i * stride + length] += 1
    covered = count > 0
    got = windows_to_points(scores, n, length, stride, "mean")
    assert np.allclose(got[covered], expected[covered] / count[covered])
    assert np.allclose(got[~covered], got[covered][-1])


def test_points_max_rule():
    scores = np.array([0.0, 5.0, 1.0])
    out = windows_to_points(scores, 8, 4, 2, "max")
    assert out.tolist() == [0.0, 0.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0]


def test_points_shape_mismatch_raises():
    with pytest.raises(ValueError):
        windows_to_points(np.zeros(3), 100, 10, 5)
