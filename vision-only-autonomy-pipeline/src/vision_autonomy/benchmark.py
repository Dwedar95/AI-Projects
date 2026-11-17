"""Reproducible synthetic benchmarks shared by the demo script and the tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import FisheyeCamera, PinholeCamera, intrinsics_from_fov
from .detection import Detection
from .metrics import TrackingMetrics, evaluate_tracking
from .pipeline import PipelineConfig, VisionPipeline
from .synthetic import (
    EgoPose,
    ObjectTruth,
    SceneConfig,
    SceneObject,
    SyntheticScene,
    default_objects,
    evaluable_truth,
    integrate_ego,
)
from .tracker import TrackerConfig, interpolate_gaps, iou_matrix

WIDTH, HEIGHT = 640, 480


def make_cameras() -> tuple[FisheyeCamera, PinholeCamera]:
    """The raw fisheye camera of the synthetic stream and the rectified pinhole target."""
    rectified = PinholeCamera(intrinsics_from_fov(WIDTH, HEIGHT, 94.0), WIDTH, HEIGHT)
    K_raw = np.array([[380.0, 0.0, 319.5], [0.0, 380.0, 239.5], [0.0, 0.0, 1.0]])
    raw = FisheyeCamera(K_raw, WIDTH, HEIGHT, np.array([-0.03, 0.004, 0.0, 0.0]))
    return raw, rectified


class DroppingDetector:
    """Wraps a detector and randomly drops detections to emulate missed detections."""

    def __init__(self, inner, drop_probability: float, seed: int = 0) -> None:
        self.inner = inner
        self.p = drop_probability
        self.rng = np.random.default_rng(seed)

    def detect(self, image: np.ndarray) -> list[Detection]:
        return [d for d in self.inner.detect(image) if self.rng.random() >= self.p]


def drop_ignored(hyp: dict[int, np.ndarray], gt: dict[int, np.ndarray],
                 objects: list[ObjectTruth], iou_thresh: float = 0.3) -> dict[int, np.ndarray]:
    """Remove hypotheses that sit on a real object excluded from the evaluation set.

    Tiny, occluded or truncated objects are not required to be tracked, but tracking
    them is not an error either, so such hypotheses are neither true nor false
    positives (the "ignore region" convention of the MOT and KITTI benchmarks).
    """
    ignored = [np.asarray(o.box) for o in objects if o.object_id not in gt]
    if not ignored or not hyp:
        return hyp
    ids = list(hyp)
    boxes = np.array([hyp[i] for i in ids])
    on_ignored = iou_matrix(boxes, np.array(ignored)).max(axis=1) >= iou_thresh
    on_gt = np.zeros(len(ids), dtype=bool)
    if gt:
        on_gt = iou_matrix(boxes, np.array(list(gt.values()))).max(axis=1) >= 0.5
    return {i: hyp[i] for i, ign, ok in zip(ids, on_ignored, on_gt, strict=True) if ok or not ign}


@dataclass
class OpenLoopReport:
    frames: int
    online: TrackingMetrics
    interpolated: TrackingMetrics
    range_abs_err_m: np.ndarray
    range_rel_err: np.ndarray
    range_true_m: np.ndarray
    speed_err: np.ndarray
    yaw_rate_err: np.ndarray
    ego_valid_ratio: float
    timings: dict[str, dict[str, float]]


def run_open_loop(
    frames: int = 150,
    dt: float = 0.1,
    speed: float = 2.0,
    yaw_amplitude: float = 0.04,
    drop_probability: float = 0.0,
    seed: int = 0,
    objects: list[SceneObject] | None = None,
    tracker: TrackerConfig | None = None,
) -> OpenLoopReport:
    """Drive a fixed ego trajectory through the synthetic scene and score every stage.

    The ego vehicle follows a gentle sinusoidal curvature profile that does not depend
    on the planner, so the result is a pure perception benchmark.
    """
    raw, rectified = make_cameras()
    scene_cfg = SceneConfig(seed=seed, objects=default_objects() if objects is None else objects)
    scene = SyntheticScene(rectified, scene_cfg, raw_camera=raw)
    pipe = VisionPipeline(raw, rectified, PipelineConfig(
        camera_height=scene_cfg.camera_height, camera_pitch=scene_cfg.camera_pitch,
        tracker=tracker or TrackerConfig()))
    if drop_probability > 0.0:
        pipe.detector = DroppingDetector(pipe.detector, drop_probability, seed)

    pose = EgoPose(0.0, 0.0, 0.0)
    gt_frames: list[dict[int, np.ndarray]] = []
    hyp_frames: list[dict[int, np.ndarray]] = []
    history: dict[int, dict[int, np.ndarray]] = {}
    truth_frames: list[list[ObjectTruth]] = []
    abs_err, rel_err, true_rng, speed_err, yaw_err = [], [], [], [], []
    ego_valid = 0
    prev_curvature = 0.0
    for i in range(frames):
        t = i * dt
        frame = scene.render(t, pose)
        result = pipe.step(frame.image, dt)
        gt = evaluable_truth(frame.objects)
        hyp = {rt.track.track_id: np.clip(rt.track.box, 0, [WIDTH - 1, HEIGHT - 1] * 2)
               for rt in result.tracks}
        for tid, box in hyp.items():
            history.setdefault(tid, {})[i] = box
        gt_frames.append(gt)
        truth_frames.append(frame.objects)
        hyp_frames.append(drop_ignored(hyp, gt, frame.objects))
        # Range error for tracks that overlap a ground-truth object.
        truth = {o.object_id: o for o in frame.objects}
        if gt and result.tracks:
            ids = list(gt)
            iou = iou_matrix(np.array([gt[g] for g in ids]),
                             np.array([rt.track.box for rt in result.tracks]))
            for gi, g in enumerate(ids):
                hi = int(np.argmax(iou[gi]))
                if iou[gi, hi] >= 0.5:
                    est = result.tracks[hi].range_m
                    abs_err.append(abs(est - truth[g].range_m))
                    rel_err.append(abs(est - truth[g].range_m) / truth[g].range_m)
                    true_rng.append(truth[g].range_m)
        curvature = yaw_amplitude * np.sin(2.0 * np.pi * t / 6.0)
        if result.ego is not None:
            ego_valid += int(result.ego.valid)
            if result.ego.valid:
                speed_err.append(abs(result.ego.speed - speed))
                yaw_err.append(abs(result.ego.yaw_rate - speed * prev_curvature))
        prev_curvature = curvature
        pose = integrate_ego(pose, speed, curvature, dt)

    filled = interpolate_gaps(history)
    interp_frames: list[dict[int, np.ndarray]] = [{} for _ in range(frames)]
    for tid, track in filled.items():
        for f, box in track.items():
            interp_frames[f][tid] = box
    interp_frames = [drop_ignored(h, g, o)
                     for h, g, o in zip(interp_frames, gt_frames, truth_frames, strict=True)]
    return OpenLoopReport(
        frames=frames,
        online=evaluate_tracking(gt_frames, hyp_frames),
        interpolated=evaluate_tracking(gt_frames, interp_frames),
        range_abs_err_m=np.asarray(abs_err), range_rel_err=np.asarray(rel_err),
        range_true_m=np.asarray(true_rng),
        speed_err=np.asarray(speed_err), yaw_rate_err=np.asarray(yaw_err),
        ego_valid_ratio=ego_valid / max(frames - 1, 1),
        timings=pipe.timer.report(skip=5),
    )


@dataclass
class ClosedLoopReport:
    frames: int
    distance_m: float
    on_road_ratio: float
    max_lateral_offset_m: float
    min_obstacle_clearance_m: float
    stops: int
    candidate_switches: int
    max_curvature_step: float


def run_closed_loop(frames: int = 200, dt: float = 0.1, seed: int = 0) -> ClosedLoopReport:
    """Let the planner drive: the commanded speed and curvature move the ego vehicle.

    The road is sinusoidal and has three parked objects on it. Ground truth from the
    scene is only used to score the run, never as planner input.
    """
    raw, rectified = make_cameras()
    objects = [
        SceneObject(1, "red", 0.9, 1.5, (16.0, 1.2)),
        SceneObject(2, "blue", 0.9, 1.5, (32.0, -1.0)),
        SceneObject(3, "yellow", 0.9, 1.2, (50.0, 2.4)),
    ]
    scene_cfg = SceneConfig(seed=seed, objects=objects, road_amplitude=2.5, road_period=70.0)
    scene = SyntheticScene(rectified, scene_cfg, raw_camera=raw)
    pipe = VisionPipeline(raw, rectified, PipelineConfig(
        camera_height=scene_cfg.camera_height, camera_pitch=scene_cfg.camera_pitch))
    pose = EgoPose(0.0, 0.0, 0.0)
    on_road = stops = switches = 0
    distance = 0.0
    max_offset = 0.0
    min_clear = np.inf
    max_step = 0.0
    last_cmd = 0.0
    for i in range(frames):
        frame = scene.render(i * dt, pose)
        result = pipe.step(frame.image, dt)
        plan = result.plan
        on_road += int(scene.on_road(pose.x, pose.y))
        max_offset = max(max_offset, abs(pose.y - float(scene.road_center(pose.x))))
        for obj in objects:
            d = float(np.hypot(obj.position[0] - pose.x, obj.position[1] - pose.y))
            min_clear = min(min_clear, d - 0.5 * obj.width)
        stops += int(plan.speed == 0.0)
        switches += int(plan.switched)
        max_step = max(max_step, abs(plan.curvature - last_cmd))
        last_cmd = plan.curvature
        distance += plan.speed * dt
        pose = integrate_ego(pose, plan.speed, plan.curvature, dt)
    return ClosedLoopReport(frames, distance, on_road / frames, max_offset, float(min_clear),
                            stops, switches, max_step)
