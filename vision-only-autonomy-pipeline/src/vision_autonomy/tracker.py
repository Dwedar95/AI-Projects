"""ByteTrack-style multi-object tracker.

Two-stage association (Zhang et al., "ByteTrack", ECCV 2022): high-confidence
detections are matched first against all active and lost tracks, then the remaining
tracks get a second chance against the low-confidence detections, which are usually
occluded or distant objects rather than clutter. Association uses IoU cost and the
Hungarian algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy.optimize import linear_sum_assignment

from .detection import Detection
from .kalman import BoxKalmanFilter, xyah_to_xyxy, xyxy_to_xyah


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of (x1, y1, x2, y2) boxes, shape (len(a), len(b))."""
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def expand_boxes(boxes: np.ndarray, ratio: np.ndarray | float) -> np.ndarray:
    """Grow (x1, y1, x2, y2) boxes by ``ratio`` times their width and height on every side."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    r = np.broadcast_to(np.asarray(ratio, dtype=np.float64), (len(b),))
    dw, dh = r * (b[:, 2] - b[:, 0]), r * (b[:, 3] - b[:, 1])
    return np.stack([b[:, 0] - dw, b[:, 1] - dh, b[:, 2] + dw, b[:, 3] + dh], axis=1)


def linear_assignment(
    cost: np.ndarray, max_cost: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian assignment with gating.

    Returns matched (row, col) pairs with cost <= max_cost plus the unmatched row and
    column indices.
    """
    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return [], list(range(n_rows)), list(range(n_cols))
    gated = np.where(cost > max_cost, max_cost + 1e3, cost)
    rows, cols = linear_sum_assignment(gated)
    matches = [(int(r), int(c)) for r, c in zip(rows, cols, strict=True) if cost[r, c] <= max_cost]
    matched_r = {r for r, _ in matches}
    matched_c = {c for _, c in matches}
    return (
        matches,
        [r for r in range(n_rows) if r not in matched_r],
        [c for c in range(n_cols) if c not in matched_c],
    )


class TrackState(Enum):
    TENTATIVE = "tentative"
    TRACKED = "tracked"
    LOST = "lost"
    REMOVED = "removed"


@dataclass
class Track:
    """One tracked object with its Kalman state and bookkeeping."""

    track_id: int
    mean: np.ndarray
    cov: np.ndarray
    score: float
    label: str
    start_frame: int
    state: TrackState = TrackState.TENTATIVE
    hits: int = 1
    last_update_frame: int = 0
    smooth_box: np.ndarray = field(default_factory=lambda: np.zeros(4))

    @property
    def box(self) -> np.ndarray:
        """Current Kalman estimate as (x1, y1, x2, y2)."""
        return xyah_to_xyxy(self.mean[:4])

    @property
    def velocity(self) -> np.ndarray:
        """Image-plane centre velocity in pixels per frame."""
        return self.mean[4:6].copy()


@dataclass(frozen=True)
class TrackOutput:
    """Snapshot of a confirmed track reported for one frame."""

    track_id: int
    box: tuple[float, float, float, float]
    score: float
    label: str
    age: int


@dataclass
class TrackerConfig:
    high_thresh: float = 0.5
    low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    first_match_cost: float = 0.8
    second_match_cost: float = 0.5
    tentative_match_cost: float = 0.7
    min_hits: int = 2
    max_time_lost: int = 30
    ema_alpha: float = 0.6
    class_aware: bool = True
    coast_frames: int = 2            # report predicted boxes for this many lost frames
    lost_buffer_rate: float = 0.08   # box growth per coasted frame for lost tracks
    lost_buffer_max: float = 0.6


class ByteTracker:
    """Online tracker; call :meth:`update` once per frame with that frame's detections."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self.kf = BoxKalmanFilter()
        self.tracks: list[Track] = []
        self.frame_id = 0
        self._next_id = 1

    def _start_track(self, det: Detection) -> Track:
        mean, cov = self.kf.initiate(xyxy_to_xyah(det.xyxy))
        track = Track(self._next_id, mean, cov, det.score, det.label, self.frame_id,
                      last_update_frame=self.frame_id, smooth_box=det.xyxy.copy())
        if self.config.min_hits <= 1:
            track.state = TrackState.TRACKED
        self._next_id += 1
        return track

    def _apply_update(self, track: Track, det: Detection) -> None:
        track.mean, track.cov = self.kf.update(track.mean, track.cov, xyxy_to_xyah(det.xyxy))
        track.score = det.score
        track.hits += 1
        track.last_update_frame = self.frame_id
        a = self.config.ema_alpha
        if track.state is TrackState.LOST:
            track.smooth_box = track.box  # restart smoothing after a gap
        else:
            track.smooth_box = a * track.box + (1.0 - a) * track.smooth_box
        if track.state is TrackState.LOST or track.hits >= self.config.min_hits:
            track.state = TrackState.TRACKED

    def _associate(
        self, tracks: list[Track], dets: list[Detection], max_cost: float
    ) -> tuple[list[Track], list[Detection]]:
        """Match and update in place; return the unmatched tracks and detections."""
        if not tracks or not dets:
            return tracks, dets
        track_boxes = np.array([t.box for t in tracks])
        det_boxes = np.array([d.xyxy for d in dets])
        cost = 1.0 - iou_matrix(track_boxes, det_boxes)
        # Buffered IoU for coasting tracks: the longer a track has been lost, the less
        # its predicted box can be trusted, so both boxes are grown before matching.
        cfg = self.config
        lost = np.array([self.frame_id - 1 - t.last_update_frame for t in tracks])
        buf = np.minimum(cfg.lost_buffer_rate * lost, cfg.lost_buffer_max)
        for i in np.nonzero(buf > 0)[0]:
            grown = iou_matrix(expand_boxes(track_boxes[i], buf[i]),
                               expand_boxes(det_boxes, buf[i]))[0]
            cost[i] = np.minimum(cost[i], 1.0 - grown)
        if self.config.class_aware:
            # A track never adopts a detection of another class.
            same = np.array([[t.label == d.label for d in dets] for t in tracks])
            cost = np.where(same, cost, np.inf)
        matches, un_t, un_d = linear_assignment(cost, max_cost)
        for ti, di in matches:
            self._apply_update(tracks[ti], dets[di])
        return [tracks[i] for i in un_t], [dets[i] for i in un_d]

    def update(self, detections: list[Detection]) -> list[TrackOutput]:
        """Advance the tracker by one frame and return the confirmed, visible tracks."""
        cfg = self.config
        self.frame_id += 1
        for t in self.tracks:
            if t.state is TrackState.LOST:
                t.mean[7] = 0.0  # do not let the box height drift while coasting
            t.mean, t.cov = self.kf.predict(t.mean, t.cov)

        high = [d for d in detections if d.score >= cfg.high_thresh]
        low = [d for d in detections if cfg.low_thresh <= d.score < cfg.high_thresh]
        confirmed = [t for t in self.tracks if t.state in (TrackState.TRACKED, TrackState.LOST)]
        tentative = [t for t in self.tracks if t.state is TrackState.TENTATIVE]

        # Stage 1: high-confidence detections against tracked and lost tracks.
        rest_tracks, rest_high = self._associate(confirmed, high, cfg.first_match_cost)
        # Stage 2: low-confidence detections against tracks that were visible last frame.
        still_tracked = [t for t in rest_tracks if t.state is TrackState.TRACKED]
        unmatched, _ = self._associate(still_tracked, low, cfg.second_match_cost)
        for t in unmatched:
            t.state = TrackState.LOST
        # Tentative tracks need a second high-confidence detection to be confirmed.
        dead, rest_high = self._associate(tentative, rest_high, cfg.tentative_match_cost)
        for t in dead:
            t.state = TrackState.REMOVED
        # Remaining confident detections start new tracks.
        for d in rest_high:
            if d.score >= cfg.new_track_thresh:
                self.tracks.append(self._start_track(d))
        for t in self.tracks:
            if t.state is TrackState.LOST and \
                    self.frame_id - t.last_update_frame > cfg.max_time_lost:
                t.state = TrackState.REMOVED
        self.tracks = [t for t in self.tracks if t.state is not TrackState.REMOVED]

        outputs = []
        for t in self.tracks:
            lost_for = self.frame_id - t.last_update_frame
            if t.state is TrackState.TRACKED and lost_for == 0:
                box = t.smooth_box
            elif t.state is TrackState.LOST and lost_for <= cfg.coast_frames:
                box = t.box  # Kalman prediction while the detector misses the object
            else:
                continue
            outputs.append(TrackOutput(t.track_id, tuple(float(v) for v in box), t.score,
                                       t.label, self.frame_id - t.start_frame + 1))
        return outputs

    def get_track(self, track_id: int) -> Track | None:
        """Look up a live track (tracked, tentative or lost) by id."""
        return next((t for t in self.tracks if t.track_id == track_id), None)


def interpolate_gaps(
    history: dict[int, dict[int, np.ndarray]], max_gap: int = 20
) -> dict[int, dict[int, np.ndarray]]:
    """Fill short detection gaps in recorded tracks by linear box interpolation.

    ``history`` maps track id to {frame: box}. This is an offline post-processing step
    (it looks ahead to the frame where the track reappears) and is therefore not part
    of the causal online output.
    """
    filled: dict[int, dict[int, np.ndarray]] = {}
    for tid, frames in history.items():
        keys = sorted(frames)
        out = {k: np.asarray(frames[k], dtype=np.float64) for k in keys}
        for k0, k1 in zip(keys[:-1], keys[1:], strict=True):
            gap = k1 - k0
            if 1 < gap <= max_gap + 1:
                for f in range(k0 + 1, k1):
                    w = (f - k0) / gap
                    out[f] = (1.0 - w) * out[k0] + w * out[k1]
        filled[tid] = dict(sorted(out.items()))
    return filled


def ema_smooth(boxes: np.ndarray, alpha: float) -> np.ndarray:
    """Causal exponential moving average over a (T, 4) box sequence."""
    boxes = np.asarray(boxes, dtype=np.float64)
    out = np.empty_like(boxes)
    for i, b in enumerate(boxes):
        out[i] = b if i == 0 else alpha * b + (1.0 - alpha) * out[i - 1]
    return out
