"""Telemetry anomaly detection for PV assets.

Two detectors are provided:

* :class:`ResidualDetector`: physics-informed. It compares measured power
  with the power the twin's PV model expects for the measured irradiance and
  cell temperature, and flags samples whose normalised residual has a large
  robust z-score (median and MAD fitted on reference data). Samples where
  both expected and measured power are near zero (night) carry no information
  and are neither used for fitting nor flagged.
* :class:`IsolationForestDetector`: model-free multivariate baseline on
  ``[irradiance, cell temperature, power]``.

:func:`inject_faults` creates labelled synthetic faults so that precision and
recall can be measured.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest

FAULT_KINDS = ("dropout", "degradation", "spike", "stuck")


class ResidualDetector:
    """Robust z-score on the residual between measured and expected power."""

    def __init__(
        self, rated_power_w: float, threshold: float = 4.0, min_active_fraction: float = 0.02
    ) -> None:
        if rated_power_w <= 0:
            raise ValueError("rated_power_w must be positive")
        self.rated_power_w = rated_power_w
        self.threshold = threshold
        self.min_active_fraction = min_active_fraction
        self._median = 0.0
        self._scale = 1.0

    def _residual(self, measured_w: np.ndarray, expected_w: np.ndarray) -> np.ndarray:
        return (np.asarray(measured_w, float) - np.asarray(expected_w, float)) / self.rated_power_w

    def _active(self, measured_w: np.ndarray, expected_w: np.ndarray) -> np.ndarray:
        level = self.min_active_fraction * self.rated_power_w
        return (np.asarray(expected_w, float) > level) | (np.asarray(measured_w, float) > level)

    def fit(self, measured_w: np.ndarray, expected_w: np.ndarray) -> ResidualDetector:
        """Estimate the residual location and scale on reference (mostly healthy) data."""
        r = self._residual(measured_w, expected_w)
        active = self._active(measured_w, expected_w)
        if active.any():
            r = r[active]
        self._median = float(np.median(r))
        mad = float(np.median(np.abs(r - self._median)))
        self._scale = max(1.4826 * mad, 1e-4)
        return self

    def score(self, measured_w: np.ndarray, expected_w: np.ndarray) -> np.ndarray:
        """Absolute robust z-score per sample (zero for inactive samples)."""
        z = np.abs(self._residual(measured_w, expected_w) - self._median) / self._scale
        return np.where(self._active(measured_w, expected_w), z, 0.0)

    def predict(self, measured_w: np.ndarray, expected_w: np.ndarray) -> np.ndarray:
        """Boolean anomaly flags."""
        return self.score(measured_w, expected_w) > self.threshold


class IsolationForestDetector:
    """Isolation Forest on raw telemetry features."""

    def __init__(self, contamination: float = 0.03, seed: int = 0) -> None:
        self.model = IsolationForest(
            n_estimators=200, contamination=contamination, random_state=seed
        )

    def fit(self, features: np.ndarray) -> IsolationForestDetector:
        """Fit on reference feature rows."""
        self.model.fit(features)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Boolean anomaly flags."""
        return self.model.predict(features) == -1


def inject_faults(
    power_w: np.ndarray,
    eligible: np.ndarray,
    n_faults: int,
    seed: int = 0,
    window: int = 6,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, str]]]:
    """Corrupt a power series with labelled faults.

    Faults start at random indices where ``eligible`` is True (for example
    daylight samples) and last ``window`` samples. Kinds: ``dropout`` (power
    reads zero), ``degradation`` (40 percent loss), ``spike`` (80 percent
    excess) and ``stuck`` (the reading freezes at its first value).

    Returns ``(corrupted, labels, events)`` where ``events`` lists the start
    index and kind of every injected fault.
    """
    rng = np.random.default_rng(seed)
    corrupted = np.asarray(power_w, dtype=float).copy()
    labels = np.zeros(len(corrupted), dtype=bool)
    candidates = np.flatnonzero(eligible[: len(corrupted) - window])
    if len(candidates) == 0 or n_faults <= 0:
        return corrupted, labels, []
    events: list[tuple[int, str]] = []
    for start in rng.choice(candidates, size=min(n_faults, len(candidates)), replace=False):
        if labels[start : start + window].any():
            continue
        kind = FAULT_KINDS[int(rng.integers(len(FAULT_KINDS)))]
        sl = slice(int(start), int(start) + window)
        if kind == "dropout":
            corrupted[sl] = 0.0
        elif kind == "degradation":
            corrupted[sl] *= 0.6
        elif kind == "spike":
            corrupted[sl] *= 1.8
        else:
            corrupted[sl] = corrupted[int(start)]
        # A stuck reading is only wrong where it differs from the true value.
        changed = np.abs(corrupted[sl] - np.asarray(power_w, float)[sl]) > 1e-9
        labels[sl] = changed if kind == "stuck" else True
        events.append((int(start), kind))
    return corrupted, labels, sorted(events)


@dataclass(frozen=True)
class DetectionScore:
    """Point-wise detection quality."""

    precision: float
    recall: float
    f1: float
    n_true: int
    n_flagged: int


def score_detection(labels: np.ndarray, flags: np.ndarray) -> DetectionScore:
    """Precision, recall and F1 of boolean flags against boolean labels."""
    labels = np.asarray(labels, dtype=bool)
    flags = np.asarray(flags, dtype=bool)
    tp = int(np.sum(labels & flags))
    n_true, n_flagged = int(labels.sum()), int(flags.sum())
    precision = tp / n_flagged if n_flagged else 0.0
    recall = tp / n_true if n_true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return DetectionScore(precision, recall, f1, n_true, n_flagged)
