"""Base anomaly detectors with one common interface.

All detectors are trained on (assumed) normal feature vectors and return a score
where a higher value means more anomalous. Isolation Forest, One-Class SVM and
Local Outlier Factor are thin wrappers around scikit-learn; the PCA reconstruction
detector is implemented in numpy.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

DETECTOR_NAMES: tuple[str, ...] = ("iforest", "ocsvm", "lof", "pca")


class Detector(Protocol):
    """Minimal detector interface used by the ensemble."""

    def fit(self, features: np.ndarray) -> Detector: ...

    def score(self, features: np.ndarray) -> np.ndarray: ...


class IsolationForestDetector:
    """Isolation Forest (Liu et al., 2008); score is the negated sklearn normality score."""

    def __init__(self, n_estimators: int = 100, seed: int = 0) -> None:
        self.model = IsolationForest(n_estimators=n_estimators, random_state=seed)

    def fit(self, features: np.ndarray) -> IsolationForestDetector:
        self.model.fit(features)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(features)


class OneClassSVMDetector:
    """One-Class SVM with RBF kernel (Schoelkopf et al., 2001)."""

    def __init__(self, nu: float = 0.05, gamma: str | float = "scale") -> None:
        self.model = OneClassSVM(nu=nu, gamma=gamma, kernel="rbf")

    def fit(self, features: np.ndarray) -> OneClassSVMDetector:
        self.model.fit(features)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        return -self.model.decision_function(features)


class LOFDetector:
    """Local Outlier Factor in novelty mode (Breunig et al., 2000)."""

    def __init__(self, n_neighbors: int = 20) -> None:
        self.n_neighbors = n_neighbors
        self.model: LocalOutlierFactor | None = None

    def fit(self, features: np.ndarray) -> LOFDetector:
        k = max(1, min(self.n_neighbors, features.shape[0] - 1))
        self.model = LocalOutlierFactor(n_neighbors=k, novelty=True)
        self.model.fit(features)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LOFDetector is not fitted")
        return -self.model.score_samples(features)


class PCAReconstructionDetector:
    """Reconstruction error on a PCA subspace of the normal data.

    The subspace keeps the smallest number of components that explain
    ``variance`` of the training variance. The score is the squared distance of a
    sample to that subspace (the Q statistic or squared prediction error).
    """

    def __init__(self, variance: float = 0.9) -> None:
        if not 0.0 < variance <= 1.0:
            raise ValueError("variance must be in (0, 1]")
        self.variance = variance
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None

    def fit(self, features: np.ndarray) -> PCAReconstructionDetector:
        x = np.asarray(features, dtype=float)
        self.mean_ = x.mean(axis=0)
        _, sing, vt = np.linalg.svd(x - self.mean_, full_matrices=False)
        ratio = np.cumsum(sing**2) / max(float((sing**2).sum()), 1e-12)
        k = int(np.searchsorted(ratio, self.variance - 1e-12) + 1)
        k = max(1, min(k, vt.shape[0] - 1 if vt.shape[0] > 1 else 1))
        self.components_ = vt[:k]
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCAReconstructionDetector is not fitted")
        centred = np.asarray(features, dtype=float) - self.mean_
        projected = centred @ self.components_.T
        residual = centred - projected @ self.components_
        return (residual**2).sum(axis=1)


def make_detector(name: str, seed: int = 0) -> Detector:
    """Factory for the detectors listed in ``DETECTOR_NAMES``."""
    if name == "iforest":
        return IsolationForestDetector(seed=seed)
    if name == "ocsvm":
        return OneClassSVMDetector()
    if name == "lof":
        return LOFDetector()
    if name == "pca":
        return PCAReconstructionDetector()
    raise ValueError(f"unknown detector '{name}', choose from {DETECTOR_NAMES}")
