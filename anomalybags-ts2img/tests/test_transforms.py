import numpy as np
import pytest

from anomalybags.transforms import (
    TRANSFORM_NAMES,
    ImageTransformer,
    gramian_angular_field,
    markov_transition_field,
    paa,
    recurrence_plot,
)


def test_paa_averages_segments():
    x = np.arange(12.0)[None, :]
    assert np.allclose(paa(x, 4), [[1.0, 4.0, 7.0, 10.0]])
    assert paa(x, 20).shape == (1, 12)


def test_gasf_matches_angular_definition(rng):
    x = rng.uniform(-1, 1, size=(3, 16))
    phi = np.arccos(x)
    expected = np.cos(phi[:, :, None] + phi[:, None, :])
    got = gramian_angular_field(x, "summation")
    assert np.allclose(got, expected)
    assert np.allclose(got, got.transpose(0, 2, 1))
    assert np.allclose(np.diagonal(got, axis1=1, axis2=2), 2 * x**2 - 1)


def test_gadf_matches_angular_definition_and_is_antisymmetric(rng):
    x = rng.uniform(-1, 1, size=(2, 10))
    phi = np.arccos(x)
    got = gramian_angular_field(x, "difference")
    assert np.allclose(got, np.sin(phi[:, :, None] - phi[:, None, :]))
    assert np.allclose(got, -got.transpose(0, 2, 1))


def test_gaf_rejects_unscaled_input():
    with pytest.raises(ValueError):
        gramian_angular_field(np.array([[0.0, 2.0]]))


def test_mtf_small_example():
    # bins: 0 0 1 1 0 -> transitions 0->0, 0->1, 1->1, 1->0, each row normalised to 0.5
    x = np.array([[0.1, 0.2, 0.8, 0.9, 0.3]])
    field = markov_transition_field(x, np.array([0.5]))
    assert field.shape == (1, 5, 5)
    assert np.allclose(field, 0.5)


def test_mtf_entries_are_probabilities(rng):
    x = rng.normal(size=(4, 40))
    edges = np.quantile(x, [0.25, 0.5, 0.75])
    field = markov_transition_field(x, edges, image_size=10)
    assert field.shape == (4, 10, 10)
    assert field.min() >= 0.0 and field.max() <= 1.0


def test_recurrence_plot_is_distance_matrix(rng):
    x = rng.normal(size=(2, 12))
    rp = recurrence_plot(x, dimension=1)
    assert np.allclose(rp, np.abs(x[:, :, None] - x[:, None, :]))
    rp2 = recurrence_plot(x, dimension=3, delay=2)
    assert rp2.shape == (2, 8, 8)
    assert np.allclose(np.diagonal(rp2, axis1=1, axis2=2), 0.0, atol=1e-6)
    binary = recurrence_plot(x, dimension=2, threshold=0.5)
    assert set(np.unique(binary)) <= {0.0, 1.0}


@pytest.mark.parametrize("name", TRANSFORM_NAMES)
def test_transformer_output_shape_and_finite(name, rng):
    series = np.sin(np.arange(600) / 7.0) + 0.1 * rng.normal(size=600)
    windows = series[:128].reshape(2, 64)
    images = ImageTransformer(name, image_size=16).fit(series).transform(windows)
    assert images.shape == (2, 16, 16)
    assert np.all(np.isfinite(images))


def test_global_scaling_keeps_level_information(rng):
    series = np.sin(np.arange(800) / 6.0) + 0.05 * rng.normal(size=800)
    window = series[:64]
    shifted = window + 0.8
    per_window = ImageTransformer("gasf", image_size=16, scale="window")
    assert np.allclose(per_window.transform(window), per_window.transform(shifted))
    fitted = ImageTransformer("gasf", image_size=16).fit(series)
    assert np.abs(fitted.transform(window) - fitted.transform(shifted)).mean() > 0.1


def test_transformer_errors():
    with pytest.raises(ValueError):
        ImageTransformer("wavelet")
    with pytest.raises(RuntimeError):
        ImageTransformer("mtf").transform(np.zeros((1, 32)))
