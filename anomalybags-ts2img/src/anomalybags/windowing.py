"""Sliding-window utilities: series to windows and window scores back to points."""

from __future__ import annotations

import numpy as np


def n_windows(n_points: int, length: int, stride: int) -> int:
    """Number of full windows of ``length`` that fit into ``n_points`` with ``stride``."""
    if length <= 0 or stride <= 0:
        raise ValueError("length and stride must be positive")
    if n_points < length:
        return 0
    return (n_points - length) // stride + 1


def sliding_windows(series: np.ndarray, length: int, stride: int = 1) -> np.ndarray:
    """Cut a 1-D series into overlapping windows.

    Returns an array of shape ``(n_windows, length)``. Trailing points that do not
    fill a complete window are dropped.
    """
    x = np.asarray(series, dtype=float)
    if x.ndim != 1:
        raise ValueError("series must be 1-D")
    count = n_windows(x.size, length, stride)
    if count == 0:
        raise ValueError(f"series of length {x.size} is shorter than window length {length}")
    view = np.lib.stride_tricks.sliding_window_view(x, length)
    return np.ascontiguousarray(view[::stride][:count])


def window_labels(labels: np.ndarray, length: int, stride: int = 1) -> np.ndarray:
    """Window-level labels: a window is anomalous if it contains any anomalous point."""
    y = np.asarray(labels).astype(int)
    count = n_windows(y.size, length, stride)
    if count == 0:
        raise ValueError("labels shorter than window length")
    view = np.lib.stride_tricks.sliding_window_view(y, length)
    return (view[::stride][:count].max(axis=1) > 0).astype(int)


def windows_to_points(
    window_scores: np.ndarray,
    n_points: int,
    length: int,
    stride: int = 1,
    reduce: str = "mean",
) -> np.ndarray:
    """Project window scores back to the time axis.

    Each point receives the mean (or max) score of all windows that cover it.
    Points not covered by any window (possible at the tail when the stride does not
    divide the series) inherit the score of the last covered point.
    """
    scores = np.asarray(window_scores, dtype=float)
    if scores.size != n_windows(n_points, length, stride):
        raise ValueError("number of window scores does not match the series geometry")
    starts = np.arange(scores.size) * stride
    if reduce == "mean":
        total = np.zeros(n_points + 1)
        count = np.zeros(n_points + 1)
        np.add.at(total, starts, scores)
        np.add.at(total, starts + length, -scores)
        np.add.at(count, starts, 1.0)
        np.add.at(count, starts + length, -1.0)
        total = np.cumsum(total)[:n_points]
        count = np.cumsum(count)[:n_points]
        covered = count > 0.5
        out = np.zeros(n_points)
        out[covered] = total[covered] / count[covered]
    elif reduce == "max":
        out = np.full(n_points, -np.inf)
        for offset in range(length):
            idx = starts + offset
            np.maximum.at(out, idx, scores)
        covered = np.isfinite(out)
    else:
        raise ValueError(f"unknown reduce rule: {reduce}")
    if not covered.all():
        last = np.flatnonzero(covered)[-1]
        out[~covered] = out[last]
    return out
