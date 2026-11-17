"""End-to-end causal pipeline: raw frame in, tracks, ranges, ego-motion and a command out."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .camera import CameraModel, PinholeCamera, Rectifier
from .detection import ColorBlobDetector, Detection, Detector
from .egomotion import EgoMotion, FlowEgoMotion, default_flow_spec
from .geometry import BevSpec, GroundProjector
from .planner import ArcRolloutPlanner, Obstacle, PlannerConfig, PlanResult
from .tracker import ByteTracker, TrackerConfig, TrackOutput
from .traversability import TraversabilityConfig, TraversabilityEstimator

STAGES = ("rectify", "detect", "track", "range", "bev", "egomotion", "traversability", "plan")


@dataclass(frozen=True)
class RangedTrack:
    """A confirmed track with its flat-ground position in the vehicle frame."""

    track: TrackOutput
    ground_xy: tuple[float, float]
    range_m: float
    width_m: float


@dataclass
class FrameResult:
    detections: list[Detection]
    tracks: list[RangedTrack]
    ego: EgoMotion | None
    traversability: np.ndarray
    plan: PlanResult
    timings_ms: dict[str, float]


@dataclass
class PipelineConfig:
    camera_height: float = 1.2
    camera_pitch: float = np.deg2rad(8.0)
    bev: BevSpec = field(default_factory=BevSpec)
    flow_bev: BevSpec = field(default_factory=default_flow_spec)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    traversability: TraversabilityConfig = field(default_factory=TraversabilityConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    max_obstacle_range: float = 15.0


class StageTimer:
    """Accumulates wall-clock time per named stage."""

    def __init__(self) -> None:
        self.samples: dict[str, list[float]] = {}
        self._t0 = 0.0

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def lap(self, name: str, into: dict[str, float]) -> None:
        now = time.perf_counter()
        ms = (now - self._t0) * 1e3
        into[name] = ms
        self.samples.setdefault(name, []).append(ms)
        self._t0 = now

    def report(self, skip: int = 0) -> dict[str, dict[str, float]]:
        """Mean and 95th percentile per stage in milliseconds, plus the frame total."""
        out: dict[str, dict[str, float]] = {}
        total = None
        for name, values in self.samples.items():
            v = np.asarray(values[skip:])
            out[name] = {"mean_ms": float(v.mean()), "p95_ms": float(np.percentile(v, 95))}
            total = v if total is None else total + v
        if total is not None:
            out["total"] = {"mean_ms": float(total.mean()),
                            "p95_ms": float(np.percentile(total, 95)),
                            "fps": float(1e3 / total.mean())}
        return out


class VisionPipeline:
    """Runs all stages on one frame at a time, using only current and past frames."""

    def __init__(self, raw_camera: CameraModel, rectified_camera: PinholeCamera,
                 config: PipelineConfig | None = None, detector: Detector | None = None) -> None:
        self.cfg = config or PipelineConfig()
        self.camera = rectified_camera
        self.rectifier = Rectifier(raw_camera, rectified_camera)
        self.detector: Detector = detector or ColorBlobDetector()
        self.tracker = ByteTracker(self.cfg.tracker)
        self.projector = GroundProjector(rectified_camera, self.cfg.camera_height,
                                         self.cfg.camera_pitch)
        self._H_bev = self.projector.bev_homography(self.cfg.bev)
        self.bev_valid = self.projector.bev_valid_mask(self.cfg.bev)
        self._H_flow = self.projector.bev_homography(self.cfg.flow_bev)
        self.egomotion = FlowEgoMotion(self.cfg.flow_bev,
                                       self.projector.bev_valid_mask(self.cfg.flow_bev))
        self.traversability = TraversabilityEstimator(self.cfg.bev, self.bev_valid,
                                                      self.cfg.traversability)
        self.planner = ArcRolloutPlanner(self.cfg.bev, self.cfg.planner)
        self.timer = StageTimer()

    def _range_tracks(self, tracks: list[TrackOutput]) -> list[RangedTrack]:
        if not tracks:
            return []
        foot = np.array([[0.5 * (t.box[0] + t.box[2]), t.box[3]] for t in tracks])
        ground, valid = self.projector.pixel_to_ground(foot)
        fx = self.camera.K[0, 0]
        out = []
        for t, g, ok in zip(tracks, ground, valid, strict=True):
            if not ok:
                continue
            rng = float(np.hypot(g[0], g[1]))
            # Metric width from the pixel width at the depth along the optical axis.
            depth = float(np.hypot(g[0], self.cfg.camera_height))
            width = (t.box[2] - t.box[0]) * depth / fx
            out.append(RangedTrack(t, (float(g[0]), float(g[1])), rng, float(width)))
        return out

    def _obstacle_mask(self, tracks: list[RangedTrack], spec: BevSpec) -> np.ndarray | None:
        """BEV cells behind each object's foot point, where IPM smears the object."""
        tracks = [rt for rt in tracks if rt.ground_xy[0] <= spec.x_max + 1.0]
        if not tracks:
            return None
        rows, cols = spec.shape
        mask = np.zeros((rows, cols), dtype=np.uint8)
        for rt in tracks:
            near = np.array(rt.ground_xy)
            far = near * (1.0 + 6.0 / max(rt.range_m, 1e-3))
            p0, p1 = spec.to_cell(np.stack([near, far]))
            thickness = int(max(3, (rt.width_m + 0.6) / spec.resolution))
            cv2.line(mask, tuple(int(v) for v in np.round(p0)), tuple(int(v) for v in np.round(p1)),
                     1, thickness)
        return mask.astype(bool)

    def step(self, raw_image: np.ndarray, dt: float, goal_curvature: float = 0.0) -> FrameResult:
        """Process one raw frame. ``dt`` is the time since the previous frame in seconds."""
        timings: dict[str, float] = {}
        self.timer.start()
        image = self.rectifier(raw_image)
        self.timer.lap("rectify", timings)
        detections = self.detector.detect(image)
        self.timer.lap("detect", timings)
        tracks = self.tracker.update(detections)
        self.timer.lap("track", timings)
        ranged = self._range_tracks(tracks)
        self.timer.lap("range", timings)
        bev = self.projector.warp_to_bev(image, self.cfg.bev, self._H_bev)
        self.timer.lap("bev", timings)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        flow_bev = self.projector.warp_to_bev(gray, self.cfg.flow_bev, self._H_flow)
        ego = self.egomotion.update(flow_bev, dt, self._obstacle_mask(ranged, self.cfg.flow_bev))
        self.timer.lap("egomotion", timings)
        trav = self.traversability.update(bev, ego)
        self.timer.lap("traversability", timings)
        obstacles = tuple(
            Obstacle(rt.ground_xy[0], rt.ground_xy[1], 0.5 * rt.width_m)
            for rt in ranged if rt.range_m <= self.cfg.max_obstacle_range
        )
        plan = self.planner.plan(trav, dt, obstacles, goal_curvature)
        self.timer.lap("plan", timings)
        return FrameResult(detections, ranged, ego, trav, plan, timings)
