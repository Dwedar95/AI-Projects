"""Flat-ground geometry: pixel back-projection to range and inverse perspective mapping.

Vehicle frame: x forward, y left, z up, origin on the ground directly below the camera.
The camera sits at ``height`` above the ground, pitched down by ``pitch`` radians.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .camera import PinholeCamera

# Camera axes (x right, y down, z forward) expressed in the vehicle frame for zero pitch.
_CAM_TO_VEHICLE = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def camera_rotation(pitch: float) -> np.ndarray:
    """Rotation taking camera-frame vectors to the vehicle frame for a downward pitch."""
    c, s = np.cos(pitch), np.sin(pitch)
    pitch_down = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return pitch_down @ _CAM_TO_VEHICLE


@dataclass
class BevSpec:
    """Metric extent and resolution of a bird's-eye-view grid in the vehicle frame.

    Row 0 is the far edge (x = x_max) and column 0 is the left edge (y = y_max), so the
    BEV image looks like a map with the vehicle at the bottom centre driving up.
    """

    x_min: float = 1.5
    x_max: float = 13.5
    y_min: float = -6.0
    y_max: float = 6.0
    resolution: float = 0.05

    @property
    def shape(self) -> tuple[int, int]:
        rows = int(round((self.x_max - self.x_min) / self.resolution))
        cols = int(round((self.y_max - self.y_min) / self.resolution))
        return rows, cols

    def ground_from_bev(self) -> np.ndarray:
        """3x3 affine matrix mapping BEV pixels (col, row, 1) to ground (x, y, 1)."""
        r = self.resolution
        return np.array([[0.0, -r, self.x_max], [-r, 0.0, self.y_max], [0.0, 0.0, 1.0]])

    def to_cell(self, xy: np.ndarray) -> np.ndarray:
        """Ground points (N x 2) to fractional BEV (col, row) coordinates."""
        xy = np.atleast_2d(xy)
        col = (self.y_max - xy[:, 1]) / self.resolution
        row = (self.x_max - xy[:, 0]) / self.resolution
        return np.stack([col, row], axis=1)

    def to_ground(self, colrow: np.ndarray) -> np.ndarray:
        """Fractional BEV (col, row) coordinates to ground points (N x 2)."""
        cr = np.atleast_2d(colrow)
        x = self.x_max - cr[:, 1] * self.resolution
        y = self.y_max - cr[:, 0] * self.resolution
        return np.stack([x, y], axis=1)


class GroundProjector:
    """Back-projects pixels of a rectified pinhole image onto the flat ground plane."""

    def __init__(self, camera: PinholeCamera, height: float, pitch: float) -> None:
        if height <= 0.0:
            raise ValueError("camera height must be positive")
        self.camera = camera
        self.height = float(height)
        self.pitch = float(pitch)
        self.R_vc = camera_rotation(pitch)
        self.cam_pos = np.array([0.0, 0.0, self.height])

    def horizon_row(self) -> float:
        """Image row of the horizon for the configured pitch."""
        return float(self.camera.K[1, 2] - self.camera.K[1, 1] * np.tan(self.pitch))

    def pixel_to_ground(self, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Intersect pixel rays with the plane z = 0.

        Returns ground points (N x 2, vehicle frame) and a validity mask. Pixels at or
        above the horizon do not hit the ground and are flagged invalid.
        """
        rays_v = self.camera.unproject(pixels) @ self.R_vc.T
        dz = rays_v[:, 2]
        valid = dz < -1e-6
        t = np.where(valid, -self.height / np.where(valid, dz, -1.0), np.nan)
        pts = self.cam_pos[None, :2] + rays_v[:, :2] * t[:, None]
        return pts, valid

    def pixel_range(self, pixels: np.ndarray) -> np.ndarray:
        """Horizontal ground range in metres from the camera foot point to each pixel."""
        pts, valid = self.pixel_to_ground(pixels)
        rng = np.hypot(pts[:, 0], pts[:, 1])
        return np.where(valid, rng, np.inf)

    def vehicle_to_pixel(self, points_vehicle: np.ndarray) -> np.ndarray:
        """Project N x 3 vehicle-frame points into the image."""
        p = np.atleast_2d(np.asarray(points_vehicle, dtype=np.float64))
        return self.camera.project((p - self.cam_pos) @ self.R_vc)

    def image_from_ground(self) -> np.ndarray:
        """Homography mapping ground coordinates (x, y, 1) to homogeneous pixels."""
        R_cv = self.R_vc.T
        t = -R_cv @ self.cam_pos
        return self.camera.K @ np.column_stack([R_cv[:, 0], R_cv[:, 1], t])

    def bev_homography(self, spec: BevSpec) -> np.ndarray:
        """Homography mapping BEV pixels to image pixels (inverse map for warping)."""
        return self.image_from_ground() @ spec.ground_from_bev()

    def warp_to_bev(self, image: np.ndarray, spec: BevSpec, H: np.ndarray | None = None,
                    interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        """Inverse perspective mapping of a rectified image onto the BEV grid."""
        if H is None:
            H = self.bev_homography(spec)
        rows, cols = spec.shape
        return cv2.warpPerspective(image, H, (cols, rows),
                                   flags=interpolation | cv2.WARP_INVERSE_MAP,
                                   borderMode=cv2.BORDER_CONSTANT)

    def bev_valid_mask(self, spec: BevSpec) -> np.ndarray:
        """Boolean BEV mask of the cells that are actually visible in the image."""
        rows, cols = spec.shape
        cc, rr = np.meshgrid(np.arange(cols), np.arange(rows))
        ground = spec.to_ground(np.stack([cc.ravel(), rr.ravel()], axis=1).astype(np.float64))
        px = self.vehicle_to_pixel(np.column_stack([ground, np.zeros(len(ground))]))
        ok = (
            np.isfinite(px[:, 0])
            & (px[:, 0] >= 0) & (px[:, 0] <= self.camera.width - 1)
            & (px[:, 1] >= 0) & (px[:, 1] <= self.camera.height - 1)
        )
        return ok.reshape(rows, cols)
