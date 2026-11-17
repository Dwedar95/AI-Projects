import numpy as np
import pytest

from anomalybags.features import (
    FeatureExtractor,
    PCAProjector,
    glcm_features,
    hog_features,
    intensity_statistics,
)


def test_intensity_statistics_known_values():
    image = np.arange(16.0).reshape(1, 4, 4)
    feats = intensity_statistics(image)
    assert feats.shape == (1, 16)
    assert feats[0, 0] == pytest.approx(7.5)
    assert feats[0, 7] == 0.0 and feats[0, 8] == 15.0
    assert feats[0, 13] == pytest.approx(1.0)  # mean |horizontal gradient|
    assert feats[0, 14] == pytest.approx(4.0)  # mean |vertical gradient|


def test_glcm_constant_image():
    feats = glcm_features(np.full((2, 8, 8), 0.3), 0.0, 1.0)
    assert feats.shape == (2, 10)
    assert np.allclose(feats[:, 0], 0.0)  # contrast
    assert np.allclose(feats[:, 1], 1.0)  # homogeneity
    assert np.allclose(feats[:, 2], 1.0)  # energy


def test_glcm_checkerboard_has_high_contrast():
    board = (np.indices((8, 8)).sum(axis=0) % 2).astype(float)[None]
    smooth = np.tile(np.linspace(0, 1, 8), (8, 1))[None]
    assert glcm_features(board, 0, 1)[0, 0] > glcm_features(smooth, 0, 1)[0, 0]


def test_hog_is_normalised_and_orientation_selective():
    ramp_x = np.tile(np.arange(12.0), (12, 1))[None]
    hog = hog_features(ramp_x, cells=2, bins=4)
    assert hog.shape == (1, 16)
    assert np.linalg.norm(hog) == pytest.approx(1.0)
    per_bin = hog.reshape(4, 4).sum(axis=0)
    assert per_bin[0] == pytest.approx(per_bin.sum())  # all votes in the 0-rad bin
    hog_y = hog_features(ramp_x.transpose(0, 2, 1), cells=2, bins=4).reshape(4, 4).sum(axis=0)
    assert hog_y[2] == pytest.approx(hog_y.sum())  # pi/2 bin


def test_pca_projector_orthonormal_and_ordered(rng):
    images = rng.normal(size=(50, 6, 6)) * np.linspace(3, 0.1, 6)[None, None, :]
    pca = PCAProjector(4).fit(images)
    assert np.allclose(pca.components_ @ pca.components_.T, np.eye(4), atol=1e-8)
    assert np.all(np.diff(pca.explained_variance_ratio_) <= 1e-12)
    assert pca.transform(images).shape == (50, 4)


def test_extractor_standardises_training_features(rng):
    images = rng.normal(size=(80, 12, 12))
    extractor = FeatureExtractor()
    feats = extractor.fit_transform(images)
    assert feats.shape == (80, 16 + 10 + 3 * 3 * 6 + 8)
    assert np.all(np.isfinite(feats))
    assert np.allclose(feats.mean(axis=0), 0.0, atol=1e-6)


def test_extractor_kind_selection_and_errors(rng):
    images = rng.normal(size=(30, 10, 10))
    assert FeatureExtractor(kinds=("hog",)).fit_transform(images).shape == (30, 54)
    with pytest.raises(ValueError):
        FeatureExtractor(kinds=("cnn",))
    with pytest.raises(RuntimeError):
        FeatureExtractor().transform(images)
