"""Bagging ensemble over transforms x detectors x bootstrap subsets.

Each ensemble member is defined by

* one time-series-to-image transform,
* one base detector,
* a bootstrap sample of the training windows (rows), and
* a random subspace of the image features (columns).

Raw detector scores live on different scales (path lengths, SVM margins, LOF
ratios, squared errors), so every member's scores are normalised against a
reference distribution before aggregation. The reference is the member's
out-of-bag training windows: windows it did not see during fitting, which avoids
the optimistic bias of scoring a detector on its own training points.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np

from anomalybags.detectors import DETECTOR_NAMES, Detector, make_detector
from anomalybags.features import FEATURE_KINDS, FeatureExtractor
from anomalybags.transforms import TRANSFORM_NAMES, ImageTransformer
from anomalybags.windowing import sliding_windows, windows_to_points

NORMALISATIONS: tuple[str, ...] = ("zscore", "robust", "minmax", "rank")
AGGREGATIONS: tuple[str, ...] = ("mean", "median", "max", "trimmed_mean")


@dataclass
class ScoreNormaliser:
    """Map raw detector scores to a common scale using reference (normal) scores."""

    method: str = "zscore"
    centre_: float = field(default=0.0, init=False)
    spread_: float = field(default=1.0, init=False)
    sorted_: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.method not in NORMALISATIONS:
            raise ValueError(f"unknown normalisation '{self.method}', choose from {NORMALISATIONS}")

    def fit(self, reference: np.ndarray) -> ScoreNormaliser:
        ref = np.asarray(reference, dtype=float)
        if ref.size == 0:
            raise ValueError("reference scores are empty")
        if self.method == "zscore":
            self.centre_, self.spread_ = float(ref.mean()), float(ref.std())
        elif self.method == "robust":
            median = float(np.median(ref))
            mad = float(np.median(np.abs(ref - median)))
            self.centre_, self.spread_ = median, 1.4826 * mad
        elif self.method == "minmax":
            self.centre_, self.spread_ = float(ref.min()), float(ref.max() - ref.min())
        else:
            self.sorted_ = np.sort(ref)
        if self.spread_ < 1e-12:
            self.spread_ = 1.0
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        s = np.asarray(scores, dtype=float)
        if self.method == "rank":
            if self.sorted_ is None:
                raise RuntimeError("ScoreNormaliser is not fitted")
            return np.searchsorted(self.sorted_, s, side="right") / self.sorted_.size
        return (s - self.centre_) / self.spread_


def aggregate_scores(member_scores: np.ndarray, rule: str = "mean") -> np.ndarray:
    """Combine a ``(n_members, n_samples)`` score matrix into one score per sample.

    ``trimmed_mean`` drops the lowest and highest 20 percent of members per sample
    (at least one on each side when there are five or more members).
    """
    m = np.atleast_2d(np.asarray(member_scores, dtype=float))
    if rule == "mean":
        return m.mean(axis=0)
    if rule == "median":
        return np.median(m, axis=0)
    if rule == "max":
        return m.max(axis=0)
    if rule == "trimmed_mean":
        cut = int(0.2 * m.shape[0])
        if cut == 0:
            return m.mean(axis=0)
        return np.sort(m, axis=0)[cut : m.shape[0] - cut].mean(axis=0)
    raise ValueError(f"unknown aggregation '{rule}', choose from {AGGREGATIONS}")


@dataclass(frozen=True)
class EnsembleConfig:
    """Hyperparameters of the bagging ensemble."""

    window: int = 64
    stride: int = 4
    image_size: int = 24
    transforms: tuple[str, ...] = TRANSFORM_NAMES
    detectors: tuple[str, ...] = DETECTOR_NAMES
    n_estimators: int = 16
    sample_frac: float = 0.8
    feature_frac: float = 0.7
    normalisation: str = "zscore"
    aggregation: str = "mean"
    feature_kinds: tuple[str, ...] = FEATURE_KINDS
    scale: str = "global"
    point_reduce: str = "mean"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        if not 0.0 < self.sample_frac <= 1.0 or not 0.0 < self.feature_frac <= 1.0:
            raise ValueError("sample_frac and feature_frac must be in (0, 1]")
        if not self.transforms or set(self.transforms) - set(TRANSFORM_NAMES):
            raise ValueError(f"transforms must be a non-empty subset of {TRANSFORM_NAMES}")
        if not self.detectors or set(self.detectors) - set(DETECTOR_NAMES):
            raise ValueError(f"detectors must be a non-empty subset of {DETECTOR_NAMES}")
        if self.normalisation not in NORMALISATIONS:
            raise ValueError(f"normalisation must be one of {NORMALISATIONS}")
        if self.aggregation not in AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {AGGREGATIONS}")


@dataclass
class Member:
    """One fitted ensemble member."""

    transform: str
    detector_name: str
    detector: Detector
    feature_idx: np.ndarray
    normaliser: ScoreNormaliser
    n_in_bag: int
    n_out_of_bag: int


class AnomalyBags:
    """Bagging ensemble of anomaly detectors on image encodings of time-series windows.

    Usage::

        model = AnomalyBags(EnsembleConfig(n_estimators=16, seed=0)).fit(train_series)
        point_scores = model.score_points(test_series)
    """

    def __init__(self, config: EnsembleConfig | None = None) -> None:
        self.config = config or EnsembleConfig()
        self.members_: list[Member] = []
        self.transformers_: dict[str, ImageTransformer] = {}
        self.extractors_: dict[str, FeatureExtractor] = {}

    def _plan(self, rng: np.random.Generator) -> list[tuple[str, str]]:
        """Choose (transform, detector) pairs: a shuffled grid, repeated if needed."""
        cfg = self.config
        grid = list(product(cfg.transforms, cfg.detectors))
        plan: list[tuple[str, str]] = []
        while len(plan) < cfg.n_estimators:
            order = rng.permutation(len(grid))
            plan.extend(grid[i] for i in order)
        return plan[: cfg.n_estimators]

    def fit(self, train_series: np.ndarray) -> AnomalyBags:
        """Fit all members on an (assumed) anomaly-free series."""
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        windows = sliding_windows(train_series, cfg.window, cfg.stride)
        n = windows.shape[0]
        if n < 20:
            raise ValueError("need at least 20 training windows")
        plan = self._plan(rng)

        features: dict[str, np.ndarray] = {}
        self.transformers_, self.extractors_ = {}, {}
        for name in sorted({t for t, _ in plan}):
            transformer = ImageTransformer(name, image_size=cfg.image_size, scale=cfg.scale)
            images = transformer.fit(train_series).transform(windows)
            extractor = FeatureExtractor(kinds=cfg.feature_kinds)
            features[name] = extractor.fit_transform(images)
            self.transformers_[name], self.extractors_[name] = transformer, extractor

        self.members_ = []
        n_draw = max(10, int(round(cfg.sample_frac * n)))
        for transform, detector_name in plan:
            feats = features[transform]
            n_feat = feats.shape[1]
            k = max(2, int(round(cfg.feature_frac * n_feat)))
            feature_idx = np.sort(rng.choice(n_feat, size=min(k, n_feat), replace=False))
            rows = rng.integers(0, n, size=n_draw)
            oob = np.setdiff1d(np.arange(n), rows)
            detector = make_detector(detector_name, seed=int(rng.integers(0, 2**31 - 1)))
            detector.fit(feats[np.ix_(rows, feature_idx)])
            reference_rows = oob if oob.size >= 10 else rows
            reference = detector.score(feats[np.ix_(reference_rows, feature_idx)])
            normaliser = ScoreNormaliser(cfg.normalisation).fit(reference)
            self.members_.append(
                Member(transform, detector_name, detector, feature_idx, normaliser,
                       int(np.unique(rows).size), int(oob.size))
            )  # fmt: skip
        return self

    def member_scores(self, series: np.ndarray) -> np.ndarray:
        """Normalised window scores of every member, shape ``(n_members, n_windows)``."""
        if not self.members_:
            raise RuntimeError("AnomalyBags is not fitted")
        cfg = self.config
        windows = sliding_windows(series, cfg.window, cfg.stride)
        features = {
            name: self.extractors_[name].transform(self.transformers_[name].transform(windows))
            for name in self.transformers_
        }
        rows = []
        for member in self.members_:
            raw = member.detector.score(features[member.transform][:, member.feature_idx])
            rows.append(member.normaliser.transform(raw))
        return np.vstack(rows)

    def score_windows(self, series: np.ndarray) -> np.ndarray:
        """Aggregated anomaly score per window (higher means more anomalous)."""
        return aggregate_scores(self.member_scores(series), self.config.aggregation)

    def score_points(self, series: np.ndarray) -> np.ndarray:
        """Aggregated anomaly score per time step."""
        cfg = self.config
        scores = self.score_windows(series)
        return windows_to_points(
            scores, np.asarray(series).size, cfg.window, cfg.stride, cfg.point_reduce
        )

    def describe(self) -> list[dict[str, object]]:
        """Human-readable summary of the fitted members."""
        return [
            {
                "transform": m.transform,
                "detector": m.detector_name,
                "n_features": int(m.feature_idx.size),
                "in_bag": m.n_in_bag,
                "out_of_bag": m.n_out_of_bag,
            }
            for m in self.members_
        ]
