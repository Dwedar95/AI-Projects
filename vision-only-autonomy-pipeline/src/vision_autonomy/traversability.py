"""LAB colour-based ground traversability on the BEV grid with temporal filtering."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .egomotion import EgoMotion
from .geometry import BevSpec


@dataclass
class TraversabilityConfig:
    ref_x: tuple[float, float] = (1.7, 3.2)      # reference patch, metres ahead
    ref_y: tuple[float, float] = (-0.6, 0.6)     # reference patch, metres lateral
    lightness_weight: float = 0.35               # L is down-weighted against shadows
    distance_threshold: float = 3.0              # normalised LAB distance at p = 0.5
    softness: float = 0.6
    model_alpha: float = 0.1                     # reference model update rate
    filter_alpha: float = 0.5                    # temporal EMA weight of the new frame
    min_std: tuple[float, float, float] = (6.0, 2.0, 2.0)
    max_patch_chroma_std: float = 6.0            # reject patch updates that are not uniform
    blur_ksize: int = 5


class TraversabilityEstimator:
    """Classifies BEV cells as drivable by LAB distance to a self-supervised ground model.

    The patch right in front of the vehicle is assumed to be drivable (the vehicle is
    about to stand on it) and provides a running mean and spread in LAB space. Each cell
    is scored by its normalised distance to that model, and the score is filtered over
    time after motion-compensating the previous estimate with the ego-motion.
    """

    def __init__(self, spec: BevSpec, valid_mask: np.ndarray,
                 config: TraversabilityConfig | None = None) -> None:
        self.spec = spec
        self.cfg = config or TraversabilityConfig()
        self.valid = valid_mask
        c0, r0 = spec.to_cell(np.array([[self.cfg.ref_x[1], self.cfg.ref_y[1]]]))[0]
        c1, r1 = spec.to_cell(np.array([[self.cfg.ref_x[0], self.cfg.ref_y[0]]]))[0]
        rows, cols = spec.shape
        self._patch = (
            slice(int(np.clip(r0, 0, rows - 1)), int(np.clip(r1, 1, rows))),
            slice(int(np.clip(c0, 0, cols - 1)), int(np.clip(c1, 1, cols))),
        )
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.state: np.ndarray | None = None

    def _update_model(self, lab: np.ndarray) -> None:
        patch = lab[self._patch].reshape(-1, 3)
        ok = self.valid[self._patch].ravel()
        if ok.sum() < 20:
            return
        patch = patch[ok]
        mean = np.median(patch, axis=0)
        std = 1.4826 * np.median(np.abs(patch - mean), axis=0)
        if self.mean is None:
            self.mean, self.std = mean, np.maximum(std, self.cfg.min_std)
            return
        if max(std[1], std[2]) > self.cfg.max_patch_chroma_std:
            return  # the patch is not a uniform surface (e.g. an obstacle entered it)
        a = self.cfg.model_alpha
        self.mean = (1 - a) * self.mean + a * mean
        self.std = np.maximum((1 - a) * self.std + a * std, self.cfg.min_std)

    def instantaneous(self, bev_bgr: np.ndarray) -> np.ndarray:
        """Single-frame traversability probability in [0, 1] for every BEV cell."""
        k = self.cfg.blur_ksize
        lab = cv2.cvtColor(cv2.blur(bev_bgr, (k, k)), cv2.COLOR_BGR2LAB).astype(np.float32)
        self._update_model(lab)
        if self.mean is None:
            return np.full(self.spec.shape, 0.5, dtype=np.float32)
        z = (lab - self.mean.astype(np.float32)) / self.std.astype(np.float32)
        z[..., 0] *= self.cfg.lightness_weight
        dist = np.sqrt(np.sum(z * z, axis=2))
        prob = 1.0 / (1.0 + np.exp((dist - self.cfg.distance_threshold) / self.cfg.softness))
        prob[~self.valid] = 0.5
        return prob.astype(np.float32)

    def _warp_previous(self, ego: EgoMotion) -> np.ndarray:
        """Move the previous estimate into the current vehicle frame."""
        c, s = np.cos(ego.dyaw), np.sin(ego.dyaw)
        # Ground point mapping current -> previous frame: p_prev = R p_cur + t.
        G = np.array([[c, -s, ego.dx], [s, c, ego.dy], [0.0, 0.0, 1.0]])
        A = self.spec.ground_from_bev()
        M = np.linalg.inv(A) @ G @ A  # current BEV pixel -> previous BEV pixel
        rows, cols = self.spec.shape
        return cv2.warpAffine(self.state, M[:2], (cols, rows),
                              flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=-1.0)

    def update(self, bev_bgr: np.ndarray, ego: EgoMotion | None = None) -> np.ndarray:
        """Process one BEV frame and return the temporally filtered probability grid."""
        prob = self.instantaneous(bev_bgr)
        if self.state is None:
            self.state = prob
            return self.state
        prev = self._warp_previous(ego) if ego is not None and ego.valid else self.state
        known = prev >= 0.0
        a = self.cfg.filter_alpha
        self.state = np.where(known, a * prob + (1 - a) * prev, prob).astype(np.float32)
        return self.state

    def mask(self, threshold: float = 0.5) -> np.ndarray:
        """Binary drivable mask from the filtered state, cleaned with a morphological open."""
        if self.state is None:
            raise RuntimeError("call update() before mask()")
        m = (self.state >= threshold).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel).astype(bool)
