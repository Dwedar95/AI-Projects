"""Vision-only autonomy pipeline: monocular perception, tracking and local planning."""

from .camera import FisheyeCamera, PinholeCamera, Rectifier, build_remap, intrinsics_from_fov
from .detection import ColorBlobDetector, Detection, Detector
from .egomotion import EgoMotion, FlowEgoMotion
from .geometry import BevSpec, GroundProjector
from .metrics import TrackingMetrics, evaluate_tracking
from .pipeline import FrameResult, PipelineConfig, VisionPipeline
from .planner import ArcRolloutPlanner, Obstacle, PlannerConfig
from .synthetic import EgoPose, SceneConfig, SceneObject, SyntheticScene
from .tracker import ByteTracker, TrackerConfig, interpolate_gaps
from .traversability import TraversabilityEstimator

__all__ = [
    "ArcRolloutPlanner", "BevSpec", "ByteTracker", "ColorBlobDetector", "Detection",
    "Detector", "EgoMotion", "EgoPose", "FisheyeCamera", "FlowEgoMotion", "FrameResult",
    "GroundProjector", "Obstacle", "PinholeCamera", "PipelineConfig", "PlannerConfig",
    "Rectifier", "SceneConfig", "SceneObject", "SyntheticScene", "TrackerConfig",
    "TrackingMetrics", "TraversabilityEstimator", "VisionPipeline", "build_remap",
    "evaluate_tracking", "interpolate_gaps", "intrinsics_from_fov",
]
__version__ = "0.1.0"
