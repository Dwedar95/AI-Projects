"""Constant-velocity Kalman filter over bounding boxes in (cx, cy, aspect, height) form."""

from __future__ import annotations

import numpy as np


def xyxy_to_xyah(box: np.ndarray) -> np.ndarray:
    """(x1, y1, x2, y2) to (centre x, centre y, width / height, height)."""
    w, h = box[2] - box[0], box[3] - box[1]
    return np.array([box[0] + 0.5 * w, box[1] + 0.5 * h, w / max(h, 1e-6), h])


def xyah_to_xyxy(xyah: np.ndarray) -> np.ndarray:
    """Inverse of :func:`xyxy_to_xyah`."""
    h = xyah[3]
    w = xyah[2] * h
    return np.array([xyah[0] - 0.5 * w, xyah[1] - 0.5 * h, xyah[0] + 0.5 * w, xyah[1] + 0.5 * h])


class BoxKalmanFilter:
    """Kalman filter with state (cx, cy, a, h, vcx, vcy, va, vh).

    Process and measurement noise scale with the box height, as in SORT / ByteTrack, so
    one set of weights works for near (large) and far (small) objects. The velocity
    weight is larger than the ByteTrack default (1/160) because the pipeline targets a
    10 Hz stream from a moving camera, where image velocities change faster per frame.
    """

    def __init__(self, std_position: float = 1.0 / 20.0, std_velocity: float = 1.0 / 40.0) -> None:
        self._F = np.eye(8)
        self._F[:4, 4:] = np.eye(4)
        self._H = np.eye(4, 8)
        self._std_pos = std_position
        self._std_vel = std_velocity

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Create a state from a first measurement with zero velocity."""
        mean = np.concatenate([measurement, np.zeros(4)])
        h = measurement[3]
        std = np.array([
            2 * self._std_pos * h, 2 * self._std_pos * h, 1e-2, 2 * self._std_pos * h,
            10 * self._std_vel * h, 10 * self._std_vel * h, 1e-5, 10 * self._std_vel * h,
        ])
        return mean, np.diag(std * std)

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Propagate the state by one frame."""
        h = mean[3]
        std = np.array([
            self._std_pos * h, self._std_pos * h, 1e-2, self._std_pos * h,
            self._std_vel * h, self._std_vel * h, 1e-5, self._std_vel * h,
        ])
        mean = self._F @ mean
        cov = self._F @ cov @ self._F.T + np.diag(std * std)
        return mean, cov

    def project(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project the state into measurement space, adding measurement noise."""
        h = mean[3]
        std = np.array([self._std_pos * h, self._std_pos * h, 1e-1, self._std_pos * h])
        return self._H @ mean, self._H @ cov @ self._H.T + np.diag(std * std)

    def update(
        self, mean: np.ndarray, cov: np.ndarray, measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Kalman correction with a new (cx, cy, a, h) measurement."""
        proj_mean, proj_cov = self.project(mean, cov)
        gain = np.linalg.solve(proj_cov, self._H @ cov).T
        mean = mean + gain @ (measurement - proj_mean)
        cov = cov - gain @ proj_cov @ gain.T
        return mean, 0.5 * (cov + cov.T)
