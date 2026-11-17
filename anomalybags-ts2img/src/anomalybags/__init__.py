"""AnomalyBags: bagging ensembles over time-series-to-image transforms.

Reference implementation by a co-author of Bayram et al. (2023), IEEE ICETAS.
It is an independent re-implementation of the idea, not the original research code.
"""

from anomalybags.detectors import DETECTOR_NAMES, make_detector
from anomalybags.ensemble import AnomalyBags, EnsembleConfig
from anomalybags.metrics import evaluate
from anomalybags.synthetic import ANOMALY_TYPES, make_benchmark
from anomalybags.transforms import TRANSFORM_NAMES, ImageTransformer

__all__ = [
    "ANOMALY_TYPES",
    "DETECTOR_NAMES",
    "TRANSFORM_NAMES",
    "AnomalyBags",
    "EnsembleConfig",
    "ImageTransformer",
    "evaluate",
    "make_benchmark",
    "make_detector",
]

__version__ = "0.1.0"
