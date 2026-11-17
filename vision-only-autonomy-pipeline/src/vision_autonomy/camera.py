"""Camera models: pinhole with Brown radial-tangential distortion and equidistant fisheye.

Conventions: the camera frame has x right, y down and z along the optical axis.
Pixels are (u, v) with u to the right and v down. All functions are vectorised over
N points and use float64.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


def intrinsics_from_fov(width: int, height: int, hfov_deg: float) -> np.ndarray:
    """Build a 3x3 intrinsic matrix for an ideal pinhole with the given horizontal FOV."""
    f = 0.5 * width / np.tan(0.5 * np.deg2rad(hfov_deg))
    return np.array([[f, 0.0, 0.5 * (width - 1)], [0.0, f, 0.5 * (height - 1)], [0.0, 0.0, 1.0]])


@dataclass
class PinholeCamera:
    """Pinhole camera with optional Brown distortion (k1, k2, p1, p2, k3)."""

    K: np.ndarray
    width: int
    height: int
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5))

    def __post_init__(self) -> None:
        self.K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        self.dist = np.asarray(self.dist, dtype=np.float64).ravel()
        if self.dist.size != 5:
            raise ValueError("Brown distortion needs 5 coefficients (k1, k2, p1, p2, k3)")

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    def _distort(self, xn: np.ndarray, yn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        k1, k2, p1, p2, k3 = self.dist
        r2 = xn * xn + yn * yn
        radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
        xd = xn * radial + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
        yd = yn * radial + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn
        return xd, yd

    def project(self, points_cam: np.ndarray) -> np.ndarray:
        """Project Nx3 camera-frame points to Nx2 pixels. Points with z <= 0 give NaN."""
        p = np.atleast_2d(np.asarray(points_cam, dtype=np.float64))
        z = p[:, 2]
        valid = z > 1e-9
        zs = np.where(valid, z, 1.0)
        xd, yd = self._distort(p[:, 0] / zs, p[:, 1] / zs)
        u = self.K[0, 0] * xd + self.K[0, 2]
        v = self.K[1, 1] * yd + self.K[1, 2]
        out = np.stack([u, v], axis=1)
        out[~valid] = np.nan
        return out

    def unproject(self, pixels: np.ndarray, iterations: int = 25) -> np.ndarray:
        """Back-project Nx2 pixels to unit bearing rays (Nx3), inverting the distortion
        with a fixed-point iteration."""
        px = np.atleast_2d(np.asarray(pixels, dtype=np.float64))
        xd = (px[:, 0] - self.K[0, 2]) / self.K[0, 0]
        yd = (px[:, 1] - self.K[1, 2]) / self.K[1, 1]
        xn, yn = xd.copy(), yd.copy()
        if np.any(self.dist != 0.0):
            for _ in range(iterations):
                xe, ye = self._distort(xn, yn)
                xn += xd - xe
                yn += yd - ye
        rays = np.stack([xn, yn, np.ones_like(xn)], axis=1)
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)


@dataclass
class FisheyeCamera:
    """Equidistant (Kannala-Brandt) fisheye.

    r = f * theta * (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8).

    The parameterisation matches ``cv2.fisheye`` so calibrations can be exchanged.
    """

    K: np.ndarray
    width: int
    height: int
    dist: np.ndarray = field(default_factory=lambda: np.zeros(4))

    def __post_init__(self) -> None:
        self.K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        self.dist = np.asarray(self.dist, dtype=np.float64).ravel()
        if self.dist.size != 4:
            raise ValueError("fisheye distortion needs 4 coefficients (k1..k4)")

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    def _theta_d(self, theta: np.ndarray) -> np.ndarray:
        k1, k2, k3, k4 = self.dist
        t2 = theta * theta
        return theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))

    def project(self, points_cam: np.ndarray) -> np.ndarray:
        """Project Nx3 camera-frame points to Nx2 pixels. Works for rays up to 90 degrees
        off axis; points behind the camera give NaN."""
        p = np.atleast_2d(np.asarray(points_cam, dtype=np.float64))
        rxy = np.hypot(p[:, 0], p[:, 1])
        theta = np.arctan2(rxy, p[:, 2])
        scale = np.where(rxy > 1e-12, self._theta_d(theta) / np.maximum(rxy, 1e-12), 1.0)
        u = self.K[0, 0] * p[:, 0] * scale + self.K[0, 2]
        v = self.K[1, 1] * p[:, 1] * scale + self.K[1, 2]
        out = np.stack([u, v], axis=1)
        out[p[:, 2] <= 0.0] = np.nan
        return out

    def unproject(self, pixels: np.ndarray, iterations: int = 10) -> np.ndarray:
        """Back-project Nx2 pixels to unit bearing rays, solving theta_d(theta) with Newton."""
        px = np.atleast_2d(np.asarray(pixels, dtype=np.float64))
        xd = (px[:, 0] - self.K[0, 2]) / self.K[0, 0]
        yd = (px[:, 1] - self.K[1, 2]) / self.K[1, 1]
        rd = np.hypot(xd, yd)
        k1, k2, k3, k4 = self.dist
        theta = rd.copy()
        for _ in range(iterations):
            t2 = theta * theta
            f = theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - rd
            df = 1.0 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))
            theta = theta - f / np.where(np.abs(df) > 1e-9, df, 1e-9)
        theta = np.clip(theta, 0.0, 0.5 * np.pi)
        s = np.where(rd > 1e-12, np.sin(theta) / np.maximum(rd, 1e-12), 1.0)
        return np.stack([xd * s, yd * s, np.cos(theta)], axis=1)


CameraModel = PinholeCamera | FisheyeCamera


def build_remap(src: CameraModel, dst: CameraModel) -> tuple[np.ndarray, np.ndarray]:
    """Lookup maps for ``cv2.remap`` that resample an image taken by ``src`` into the
    image ``dst`` would have taken from the same position.

    For every destination pixel the bearing ray is computed with ``dst.unproject`` and
    projected into the source image. Use it with (fisheye -> ideal pinhole) to rectify a
    stream, or with (pinhole -> fisheye) to synthesise a distorted image.
    """
    w, h = dst.size
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    rays = dst.unproject(np.stack([uu.ravel(), vv.ravel()], axis=1))
    src_px = src.project(rays)
    src_px = np.where(np.isfinite(src_px), src_px, -1.0)
    map_x = src_px[:, 0].reshape(h, w).astype(np.float32)
    map_y = src_px[:, 1].reshape(h, w).astype(np.float32)
    return map_x, map_y


class Rectifier:
    """Precomputed undistortion of a raw stream into an ideal pinhole image."""

    def __init__(self, raw: CameraModel, rectified: PinholeCamera) -> None:
        self.raw = raw
        self.rectified = rectified
        self._map_x, self._map_y = build_remap(raw, rectified)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return cv2.remap(image, self._map_x, self._map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)


def checkerboard_points(cols: int, rows: int, square: float) -> np.ndarray:
    """Planar calibration target corners (N x 3, z = 0) in target coordinates."""
    xs, ys = np.meshgrid(np.arange(cols), np.arange(rows))
    return np.stack([xs.ravel() * square, ys.ravel() * square, np.zeros(cols * rows)], axis=1)


def look_at_target_pose(rvec: np.ndarray, tvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation matrix and translation of a target pose given as a Rodrigues vector."""
    rot, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    return rot, np.asarray(tvec, dtype=np.float64).reshape(3)


def calibrate_pinhole(
    object_points: list[np.ndarray], image_points: list[np.ndarray], size: tuple[int, int]
) -> tuple[PinholeCamera, float]:
    """Estimate intrinsics and Brown distortion from planar target views.

    Thin wrapper around ``cv2.calibrateCamera``. Returns the camera and the RMS
    reprojection error in pixels.
    """
    obj = [np.asarray(o, dtype=np.float32).reshape(-1, 1, 3) for o in object_points]
    img = [np.asarray(i, dtype=np.float32).reshape(-1, 1, 2) for i in image_points]
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, img, size, None, None)
    return PinholeCamera(K, size[0], size[1], dist.ravel()[:5]), float(rms)


def reprojection_rmse(
    camera: CameraModel, points_cam: np.ndarray, observed_px: np.ndarray
) -> float:
    """Root-mean-square pixel error between projected points and observations."""
    err = camera.project(points_cam) - np.asarray(observed_px, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(err * err, axis=1))))
