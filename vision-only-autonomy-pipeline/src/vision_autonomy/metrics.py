"""Multi-object tracking metrics: CLEAR-MOT (MOTA, MOTP, ID switches) and IDF1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .tracker import iou_matrix

Boxes = dict[int, np.ndarray]


@dataclass(frozen=True)
class TrackingMetrics:
    """Aggregated tracking scores over a sequence."""

    num_gt: int
    true_positives: int
    false_positives: int
    false_negatives: int
    id_switches: int
    mota: float
    motp_iou: float
    precision: float
    recall: float
    idf1: float
    idtp: int

    def summary(self) -> str:
        return (
            f"MOTA={self.mota:.3f} IDF1={self.idf1:.3f} IDSW={self.id_switches} "
            f"recall={self.recall:.3f} precision={self.precision:.3f} "
            f"MOTP(IoU)={self.motp_iou:.3f} GT={self.num_gt}"
        )


def _match_frame(
    gt: Boxes, hyp: Boxes, iou_thresh: float, previous: dict[int, int]
) -> dict[int, tuple[int, float]]:
    """CLEAR-MOT frame matching: keep still-valid previous pairs, then Hungarian."""
    gt_ids, hyp_ids = list(gt), list(hyp)
    if not gt_ids or not hyp_ids:
        return {}
    iou = iou_matrix(np.array([gt[g] for g in gt_ids]), np.array([hyp[h] for h in hyp_ids]))
    matches: dict[int, tuple[int, float]] = {}
    used_h: set[int] = set()
    for gi, g in enumerate(gt_ids):
        h = previous.get(g)
        if h in hyp and h not in used_h:
            hi = hyp_ids.index(h)
            if iou[gi, hi] >= iou_thresh:
                matches[g] = (h, float(iou[gi, hi]))
                used_h.add(h)
    free_g = [i for i, g in enumerate(gt_ids) if g not in matches]
    free_h = [i for i, h in enumerate(hyp_ids) if h not in used_h]
    if free_g and free_h:
        sub = iou[np.ix_(free_g, free_h)]
        rows, cols = linear_sum_assignment(np.where(sub >= iou_thresh, 1.0 - sub, 1e3))
        for r, c in zip(rows, cols, strict=True):
            if sub[r, c] >= iou_thresh:
                matches[gt_ids[free_g[r]]] = (hyp_ids[free_h[c]], float(sub[r, c]))
    return matches


def evaluate_tracking(
    gt_frames: list[Boxes], hyp_frames: list[Boxes], iou_thresh: float = 0.5
) -> TrackingMetrics:
    """Score hypothesis tracks against ground truth.

    Both inputs are per-frame dicts mapping an identity to an (x1, y1, x2, y2) box.
    MOTA = 1 - (FN + FP + IDSW) / GT. IDF1 uses the identity-level bipartite matching
    of Ristani et al. (2016) that maximises the number of identity true positives.
    """
    if len(gt_frames) != len(hyp_frames):
        raise ValueError("ground truth and hypotheses must cover the same frames")
    tp = fp = fn = idsw = num_gt = 0
    iou_sum = 0.0
    last_match: dict[int, int] = {}
    overlap: dict[tuple[int, int], int] = {}
    gt_count: dict[int, int] = {}
    hyp_count: dict[int, int] = {}
    for gt, hyp in zip(gt_frames, hyp_frames, strict=True):
        num_gt += len(gt)
        for g in gt:
            gt_count[g] = gt_count.get(g, 0) + 1
        for h in hyp:
            hyp_count[h] = hyp_count.get(h, 0) + 1
        matches = _match_frame(gt, hyp, iou_thresh, last_match)
        for g, (h, iou) in matches.items():
            if g in last_match and last_match[g] != h:
                idsw += 1
            last_match[g] = h
            iou_sum += iou
        tp += len(matches)
        fn += len(gt) - len(matches)
        fp += len(hyp) - len(matches)
        # Identity overlap counts for IDF1 (every pair that overlaps, not only the match).
        if gt and hyp:
            g_ids, h_ids = list(gt), list(hyp)
            iou = iou_matrix(np.array([gt[g] for g in g_ids]), np.array([hyp[h] for h in h_ids]))
            for gi, hi in zip(*np.nonzero(iou >= iou_thresh), strict=True):
                key = (g_ids[gi], h_ids[hi])
                overlap[key] = overlap.get(key, 0) + 1

    idtp = 0
    if overlap:
        g_ids, h_ids = sorted(gt_count), sorted(hyp_count)
        weight = np.zeros((len(g_ids), len(h_ids)))
        for (g, h), n in overlap.items():
            weight[g_ids.index(g), h_ids.index(h)] = n
        rows, cols = linear_sum_assignment(-weight)
        idtp = int(weight[rows, cols].sum())
    total_hyp = sum(hyp_count.values())
    idf1 = 2.0 * idtp / (num_gt + total_hyp) if (num_gt + total_hyp) else 0.0
    return TrackingMetrics(
        num_gt=num_gt,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        id_switches=idsw,
        mota=1.0 - (fn + fp + idsw) / num_gt if num_gt else 0.0,
        motp_iou=iou_sum / tp if tp else 0.0,
        precision=tp / (tp + fp) if (tp + fp) else 0.0,
        recall=tp / (tp + fn) if (tp + fn) else 0.0,
        idf1=idf1,
        idtp=idtp,
    )
