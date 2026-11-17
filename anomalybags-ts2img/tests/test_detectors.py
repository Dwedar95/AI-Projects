import numpy as np
import pytest

from anomalybags.detectors import DETECTOR_NAMES, PCAReconstructionDetector, make_detector


@pytest.mark.parametrize("name", DETECTOR_NAMES)
def test_detectors_rank_outliers_above_inliers(name, rng):
    latent = rng.normal(size=(300, 2))
    mixing = rng.normal(size=(2, 6))
    normal = latent @ mixing + 0.05 * rng.normal(size=(300, 6))
    inliers = rng.normal(size=(40, 2)) @ mixing + 0.05 * rng.normal(size=(40, 6))
    outliers = rng.normal(size=(40, 6)) * 4.0
    detector = make_detector(name, seed=0).fit(normal)
    assert detector.score(outliers).mean() > detector.score(inliers).mean()
    assert np.median(detector.score(outliers)) > np.quantile(detector.score(inliers), 0.9)


def test_pca_reconstruction_zero_error_inside_subspace(rng):
    basis = rng.normal(size=(2, 5))
    train = rng.normal(size=(100, 2)) @ basis
    detector = PCAReconstructionDetector(variance=0.999).fit(train)
    inside = rng.normal(size=(10, 2)) @ basis
    assert np.allclose(detector.score(inside), 0.0, atol=1e-8)
    assert detector.score(rng.normal(size=(10, 5))).min() > 1e-3


def test_unknown_detector_and_unfitted_errors():
    with pytest.raises(ValueError):
        make_detector("autoencoder")
    with pytest.raises(RuntimeError):
        PCAReconstructionDetector().score(np.zeros((2, 3)))
