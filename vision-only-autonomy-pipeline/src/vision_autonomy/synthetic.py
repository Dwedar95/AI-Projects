"""Synthetic driving scenes with exact ground truth.

Renders a textured road with grass shoulders and moving upright colour billboards as
seen from a forward-looking camera on a moving vehicle. Everything is synthetic: the
purpose is to measure tracking accuracy, flat-ground range error and ego-motion error
against known ground truth, not to look realistic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .camera import CameraModel, PinholeCamera, build_remap
from .geometry import camera_rotation

OBJECT_COLORS_BGR = {
    "red": (40, 40, 225),
    "yellow": (30, 215, 235),
    "blue": (225, 90, 30),
    "magenta": (215, 40, 215),
}


@dataclass
class SceneObject:
    """Upright rectangle standing on the ground, moving at constant world velocity."""

    object_id: int
    label: str
    width: float
    height: float
    position: tuple[float, float]
    velocity: tuple[float, float] = (0.0, 0.0)

    def position_at(self, t: float) -> np.ndarray:
        return np.array([self.position[0] + self.velocity[0] * t,
                         self.position[1] + self.velocity[1] * t])


@dataclass(frozen=True)
class EgoPose:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class ObjectTruth:
    """Ground truth of one object in one frame (rectified image coordinates)."""

    object_id: int
    label: str
    box: tuple[float, float, float, float]
    range_m: float
    position_vehicle: tuple[float, float]
    visible_fraction: float
    in_image_fraction: float


@dataclass
class Frame:
    image: np.ndarray
    time: float
    pose: EgoPose
    objects: list[ObjectTruth]


@dataclass
class SceneConfig:
    camera_height: float = 1.2
    camera_pitch: float = np.deg2rad(8.0)
    road_half_width: float = 3.5
    road_amplitude: float = 0.0          # lateral amplitude of a sinusoidal road, metres
    road_period: float = 60.0
    noise_sigma: float = 2.0             # additive sensor noise, grey levels
    max_object_range: float = 40.0
    seed: int = 0
    objects: list[SceneObject] = field(default_factory=list)


def _smooth_noise(rng: np.random.Generator, size: int, cell: int) -> np.ndarray:
    """Tileable band-limited noise in [-1, 1] with features about ``cell`` pixels wide."""
    coarse = rng.uniform(-1.0, 1.0, (size // cell, size // cell)).astype(np.float32)
    padded = np.pad(coarse, 1, mode="wrap")
    up = cv2.resize(padded, ((size // cell + 2) * cell,) * 2, interpolation=cv2.INTER_CUBIC)
    return up[cell:-cell, cell:-cell]


class SyntheticScene:
    """Renders frames for arbitrary ego poses; the caller owns the ego trajectory."""

    TEX_RES = 0.05  # metres per texture pixel
    TEX_SIZE = 512

    def __init__(self, camera: PinholeCamera, config: SceneConfig,
                 raw_camera: CameraModel | None = None) -> None:
        self.camera = camera
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)
        self._octaves = [_smooth_noise(self.rng, self.TEX_SIZE, c) for c in (2, 8, 32)]
        self.R_vc = camera_rotation(config.camera_pitch)
        self.cam_pos = np.array([0.0, 0.0, config.camera_height])
        w, h = camera.size
        uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        rays = camera.unproject(np.stack([uu.ravel(), vv.ravel()], axis=1)) @ self.R_vc.T
        self._ground_idx = np.nonzero(rays[:, 2] < -1e-3)[0]
        t = -config.camera_height / rays[self._ground_idx, 2]
        self._ground_xy = rays[self._ground_idx, :2] * t[:, None]
        rng_m = np.hypot(self._ground_xy[:, 0], self._ground_xy[:, 1])
        # Fade fine texture octaves with range to avoid aliasing far from the camera.
        self._octave_weight = [
            np.clip(2.0 / rng_m, 0.0, 1.0), np.clip(8.0 / rng_m, 0.0, 1.0),
            np.clip(40.0 / rng_m, 0.0, 1.0),
        ]
        self._distort = build_remap(camera, raw_camera) if raw_camera is not None else None

    def road_center(self, x_world: np.ndarray | float) -> np.ndarray | float:
        """Lateral position of the road centre line at a given world x."""
        return self.cfg.road_amplitude * np.sin(2.0 * np.pi * np.asarray(x_world)
                                                / self.cfg.road_period)

    def on_road(self, x_world: float, y_world: float) -> bool:
        return abs(y_world - float(self.road_center(x_world))) <= self.cfg.road_half_width

    def _render_ground(self, pose: EgoPose) -> np.ndarray:
        w, h = self.camera.size
        c, s = np.cos(pose.yaw), np.sin(pose.yaw)
        xw = pose.x + c * self._ground_xy[:, 0] - s * self._ground_xy[:, 1]
        yw = pose.y + s * self._ground_xy[:, 0] + c * self._ground_xy[:, 1]
        ix = np.floor(xw / self.TEX_RES).astype(np.int64) % self.TEX_SIZE
        iy = np.floor(yw / self.TEX_RES).astype(np.int64) % self.TEX_SIZE
        noise = sum(wt * oc[iy, ix] for wt, oc in zip(self._octave_weight, self._octaves,
                                                      strict=True)) / 3.0
        road = np.abs(yw - self.road_center(xw)) <= self.cfg.road_half_width
        base = np.where(road[:, None], np.array([98.0, 96.0, 94.0]), np.array([62.0, 118.0, 72.0]))
        ground = base * (1.0 + 0.45 * noise[:, None])
        img = np.empty((h * w, 3), dtype=np.float32)
        img[:] = (205.0, 185.0, 175.0)  # sky, low saturation
        img[self._ground_idx] = ground
        return img.reshape(h, w, 3)

    def _object_truth(self, obj: SceneObject, t: float, pose: EgoPose):
        """Project one object; returns (truth fields, polygon) or None if not in view."""
        c, s = np.cos(pose.yaw), np.sin(pose.yaw)
        d = obj.position_at(t) - np.array([pose.x, pose.y])
        pv = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
        rng_m = float(np.hypot(pv[0], pv[1]))
        if pv[0] < 0.8 or rng_m > self.cfg.max_object_range:
            return None
        lateral = np.array([-pv[1], pv[0]]) / rng_m  # billboard faces the camera
        half = 0.5 * obj.width * lateral
        corners = np.array([
            [*(pv - half), 0.0], [*(pv + half), 0.0],
            [*(pv + half), obj.height], [*(pv - half), obj.height],
        ])
        px = self.camera.project((corners - self.cam_pos) @ self.R_vc)
        if not np.all(np.isfinite(px)):
            return None
        w, h = self.camera.size
        x1, y1 = px.min(axis=0)
        x2, y2 = px.max(axis=0)
        cx1, cy1 = np.clip(x1, 0, w - 1), np.clip(y1, 0, h - 1)
        cx2, cy2 = np.clip(x2, 0, w - 1), np.clip(y2, 0, h - 1)
        full_area = max((x2 - x1) * (y2 - y1), 1e-9)
        in_frac = float((cx2 - cx1) * (cy2 - cy1) / full_area)
        if in_frac <= 0.0:
            return None
        return (rng_m, pv, (float(cx1), float(cy1), float(cx2), float(cy2)), in_frac, px)

    def render(self, t: float, pose: EgoPose) -> Frame:
        """Render the scene at time ``t`` from ``pose`` and return image plus ground truth."""
        img = self._render_ground(pose)
        h, w = img.shape[:2]
        visible = []
        for obj in self.cfg.objects:
            info = self._object_truth(obj, t, pose)
            if info is not None:
                visible.append((obj, *info))
        visible.sort(key=lambda item: -item[1])  # painter's algorithm, far to near
        id_buffer = np.zeros((h, w), dtype=np.int32)
        own_area: dict[int, int] = {}
        for obj, _rng, _pv, _box, _frac, px in visible:
            poly = np.round(px).astype(np.int32)
            cv2.fillConvexPoly(img, poly, OBJECT_COLORS_BGR[obj.label])
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, poly, 1)
            own_area[obj.object_id] = int(mask.sum())
            id_buffer[mask > 0] = obj.object_id
        truths = []
        for obj, rng_m, pv, box, in_frac, _px in visible:
            area = own_area[obj.object_id]
            vis = float((id_buffer == obj.object_id).sum() / area) if area else 0.0
            truths.append(ObjectTruth(obj.object_id, obj.label, box, rng_m,
                                      (float(pv[0]), float(pv[1])), vis, in_frac))
        if self.cfg.noise_sigma > 0:
            img += self.rng.normal(0.0, self.cfg.noise_sigma, img.shape).astype(np.float32)
        out = np.clip(img, 0, 255).astype(np.uint8)
        if self._distort is not None:
            out = cv2.remap(out, self._distort[0], self._distort[1], cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT)
        return Frame(out, t, pose, truths)


def evaluable_truth(objects: list[ObjectTruth], min_area: float = 150.0,
                    min_visible: float = 0.5, min_in_image: float = 0.6) -> dict[int, np.ndarray]:
    """Ground-truth boxes that a detector can reasonably be expected to find.

    Objects that are tiny, mostly occluded or mostly outside the image are excluded
    from the evaluation set. The thresholds are part of the reported protocol.
    """
    out: dict[int, np.ndarray] = {}
    for o in objects:
        area = (o.box[2] - o.box[0]) * (o.box[3] - o.box[1])
        if area >= min_area and o.visible_fraction >= min_visible \
                and o.in_image_fraction >= min_in_image:
            out[o.object_id] = np.asarray(o.box)
    return out


def integrate_ego(pose: EgoPose, speed: float, curvature: float, dt: float) -> EgoPose:
    """Exact unicycle integration of the ego pose over ``dt`` at constant speed/curvature."""
    dyaw = speed * curvature * dt
    if abs(dyaw) < 1e-9:
        return EgoPose(pose.x + speed * dt * np.cos(pose.yaw),
                       pose.y + speed * dt * np.sin(pose.yaw), pose.yaw)
    r = 1.0 / curvature
    return EgoPose(
        pose.x + r * (np.sin(pose.yaw + dyaw) - np.sin(pose.yaw)),
        pose.y - r * (np.cos(pose.yaw + dyaw) - np.cos(pose.yaw)),
        pose.yaw + dyaw,
    )


def default_objects() -> list[SceneObject]:
    """A fixed mix of oncoming, crossing, slower and parked objects for the benchmark."""
    return [
        SceneObject(1, "red", 0.8, 1.7, (14.0, 1.6), (1.2, 0.0)),
        SceneObject(2, "blue", 1.6, 1.4, (30.0, -1.8), (-1.5, 0.0)),
        SceneObject(3, "yellow", 0.6, 1.1, (18.0, -5.0), (1.4, 0.75)),
        SceneObject(4, "magenta", 0.7, 1.8, (26.0, 5.5), (1.0, -0.6)),
        SceneObject(5, "red", 1.8, 1.5, (55.0, 2.0), (-0.5, 0.0)),
        SceneObject(6, "blue", 0.7, 1.7, (40.0, -2.6), (1.6, 0.0)),
        SceneObject(7, "yellow", 1.0, 1.0, (48.0, 0.5), (0.0, 0.0)),
        SceneObject(8, "magenta", 0.8, 1.6, (36.0, -6.0), (0.9, 0.9)),
    ]
