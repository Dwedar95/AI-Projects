"""Ego-motion from Farneback dense optical flow on the bird's-eye-view image.

On the IPM image a flat, static ground moves as a rigid 2D transform, so the dense
flow field can be reduced to a planar ego-motion (forward, lateral, yaw) with a robust
rigid fit. Anything that does not move with the ground (objects, IPM stretch
artefacts) is rejected as an outlier.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import BevSpec


@dataclass(frozen=True)
class EgoMotion:
    """Planar ego-motion between two frames, expressed in the earlier vehicle frame."""

    dx: float
    dy: float
    dyaw: float
    speed: float
    yaw_rate: float
    inlier_ratio: float
    valid: bool


def fit_rigid_2d(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares rotation R and translation t with dst ~= src @ R.T + t (Kabsch)."""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    cov = (dst - mu_d).T @ (src - mu_s)
    u, _, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(u @ vt))
    R = u @ np.diag([1.0, d]) @ vt
    return R, mu_d - R @ mu_s


def robust_rigid_2d(
    src: np.ndarray, dst: np.ndarray, iterations: int = 3, k_mad: float = 3.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rigid fit with iterative MAD-based outlier rejection. Returns R, t, inlier mask."""
    # Start from the median displacement so a minority of moving points cannot bias it.
    R, t = np.eye(2), np.median(dst - src, axis=0)
    inliers = np.ones(len(src), dtype=bool)
    for _ in range(iterations):
        res = np.linalg.norm(src @ R.T + t - dst, axis=1)
        med = np.median(res[inliers])
        mad = np.median(np.abs(res[inliers] - med)) + 1e-9
        inliers = res <= med + k_mad * 1.4826 * mad
        if inliers.sum() < 3:
            break
        R, t = fit_rigid_2d(src[inliers], dst[inliers])
    return R, t, inliers


def default_flow_spec() -> BevSpec:
    """Near-field grid used for flow: fully visible and well sampled by the camera."""
    return BevSpec(x_min=1.4, x_max=4.4, y_min=-1.6, y_max=1.6, resolution=0.04)


class FlowEgoMotion:
    """Estimates planar ego-motion from consecutive BEV grayscale images.

    The grid should cover only the near field. Far from the camera one image pixel
    spans many BEV cells, so the IPM image there is an interpolation of few samples and
    its apparent motion is unreliable.
    """

    def __init__(self, spec: BevSpec, valid_mask: np.ndarray | None = None, stride: int = 4,
                 border: int = 8, min_points: int = 40) -> None:
        self.spec = spec
        self.min_points = min_points
        rows, cols = spec.shape
        rr, cc = np.mgrid[border:rows - border:stride, border:cols - border:stride]
        rr, cc = rr.ravel(), cc.ravel()
        if valid_mask is not None:
            keep = valid_mask[rr, cc]
            rr, cc = rr[keep], cc[keep]
        self._rr, self._cc = rr, cc
        self._ground = spec.to_ground(np.stack([cc, rr], axis=1).astype(np.float64))
        self._prev: np.ndarray | None = None

    def reset(self) -> None:
        self._prev = None

    def update(self, bev_gray: np.ndarray, dt: float,
               ignore_mask: np.ndarray | None = None) -> EgoMotion | None:
        """Feed the next BEV image; returns None for the first frame.

        ``ignore_mask`` marks BEV cells (for example non-ground obstacles) to exclude.
        """
        prev, self._prev = self._prev, bev_gray
        if prev is None:
            return None
        flow = cv2.calcOpticalFlowFarneback(
            prev, bev_gray, None, pyr_scale=0.5, levels=3, winsize=15, iterations=8,
            poly_n=5, poly_sigma=1.1, flags=0,
        )
        f = flow[self._rr, self._cc].astype(np.float64)
        keep = np.ones(len(f), dtype=bool)
        if ignore_mask is not None:
            keep &= ~ignore_mask[self._rr, self._cc]
        # The flow target must stay inside the image, otherwise it is extrapolated.
        rows, cols = self.spec.shape
        tc, tr = self._cc + f[:, 0], self._rr + f[:, 1]
        keep &= (tc >= 0) & (tc <= cols - 1) & (tr >= 0) & (tr <= rows - 1)
        if keep.sum() < self.min_points:
            return EgoMotion(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False)
        src = self._ground[keep]
        res = self.spec.resolution
        dst = src + np.stack([-f[keep, 1] * res, -f[keep, 0] * res], axis=1)
        R, t, inliers = robust_rigid_2d(src, dst)
        # Static ground obeys dst = R_ego^T (src - t_ego), hence the inversion below.
        dyaw = -float(np.arctan2(R[1, 0], R[0, 0]))
        trans = -R.T @ t
        ratio = float(inliers.mean())
        return EgoMotion(
            dx=float(trans[0]), dy=float(trans[1]), dyaw=dyaw,
            speed=float(np.hypot(trans[0], trans[1]) / dt), yaw_rate=dyaw / dt,
            inlier_ratio=ratio, valid=bool(inliers.sum() >= self.min_points),
        )
