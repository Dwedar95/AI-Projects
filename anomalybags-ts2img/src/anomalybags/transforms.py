"""Time-series-to-image transforms implemented in numpy.

Implemented transforms (all batched over windows):

* ``gasf`` / ``gadf``: Gramian Angular Summation / Difference Field (Wang and Oates, 2015)
* ``mtf``: Markov Transition Field (Wang and Oates, 2015)
* ``rp``: Recurrence Plot (Eckmann et al., 1987), unthresholded by default

Scaling matters for anomaly detection. The textbook Gramian field rescales every
window to [-1, 1] on its own, which removes amplitude information: a level shift or
a variance change that fills a whole window becomes invisible. ``ImageTransformer``
therefore fits its scaling (value range, quantile bin edges, distance scale) on the
anomaly-free training series by default (``scale="global"``) and offers the
per-window textbook behaviour as ``scale="window"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TRANSFORM_NAMES: tuple[str, ...] = ("gasf", "gadf", "mtf", "rp")


def paa_matrix(length: int, size: int) -> np.ndarray:
    """Averaging matrix ``A`` of shape ``(size, length)`` for Piecewise Aggregate Approximation.

    ``A @ x`` gives the mean of ``x`` over ``size`` nearly equal consecutive segments.
    """
    if size <= 0 or size > length:
        raise ValueError("size must be in [1, length]")
    segment = (np.arange(length) * size) // length
    mat = np.zeros((size, length))
    mat[segment, np.arange(length)] = 1.0
    return mat / mat.sum(axis=1, keepdims=True)


def paa(windows: np.ndarray, size: int) -> np.ndarray:
    """Piecewise Aggregate Approximation of each row of ``windows`` to ``size`` points."""
    x = np.atleast_2d(np.asarray(windows, dtype=float))
    if size >= x.shape[1]:
        return x
    return x @ paa_matrix(x.shape[1], size).T


def scale_to_unit(
    windows: np.ndarray, lo: np.ndarray | float, hi: np.ndarray | float
) -> np.ndarray:
    """Affine map of ``[lo, hi]`` to ``[-1, 1]`` with clipping of out-of-range values."""
    span = np.maximum(np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float), 1e-12)
    return np.clip(2.0 * (windows - lo) / span - 1.0, -1.0, 1.0)


def gramian_angular_field(scaled: np.ndarray, kind: str = "summation") -> np.ndarray:
    """Gramian Angular Field of rows already scaled to [-1, 1].

    With ``phi = arccos(x)``: GASF[i, j] = cos(phi_i + phi_j) and
    GADF[i, j] = sin(phi_i - phi_j). Both are evaluated with the closed forms
    in ``x`` and ``sqrt(1 - x^2)`` so no trigonometric call is needed.
    """
    x = np.atleast_2d(np.asarray(scaled, dtype=float))
    if np.any(np.abs(x) > 1.0 + 1e-9):
        raise ValueError("input must be scaled to [-1, 1]")
    x = np.clip(x, -1.0, 1.0)
    s = np.sqrt(1.0 - x**2)
    if kind == "summation":
        return x[:, :, None] * x[:, None, :] - s[:, :, None] * s[:, None, :]
    if kind == "difference":
        return s[:, :, None] * x[:, None, :] - x[:, :, None] * s[:, None, :]
    raise ValueError(f"unknown Gramian field kind: {kind}")


def markov_transition_field(
    windows: np.ndarray, bin_edges: np.ndarray, image_size: int | None = None
) -> np.ndarray:
    """Markov Transition Field of each row of ``windows``.

    ``bin_edges`` are the interior edges (``n_bins - 1`` values, shared by all rows or
    one row of edges per window). Each window gets its own first-order transition
    matrix ``W`` (row-normalised), and ``MTF[i, j] = W[bin(x_i), bin(x_j)]``. The full
    field is reduced to ``image_size`` by block averaging.
    """
    x = np.atleast_2d(np.asarray(windows, dtype=float))
    n, length = x.shape
    edges = np.asarray(bin_edges, dtype=float)
    if edges.ndim == 1:
        bins = np.searchsorted(edges, x, side="right")
        n_bins = edges.size + 1
    else:
        bins = (x[:, :, None] >= edges[:, None, :]).sum(axis=2)
        n_bins = edges.shape[1] + 1
    code = bins[:, :-1] * n_bins + bins[:, 1:] + (np.arange(n) * n_bins * n_bins)[:, None]
    counts = np.bincount(code.ravel(), minlength=n * n_bins * n_bins).astype(float)
    trans = counts.reshape(n, n_bins, n_bins)
    row_sum = trans.sum(axis=2, keepdims=True)
    trans = np.divide(trans, row_sum, out=np.zeros_like(trans), where=row_sum > 0)
    rows = np.arange(n)[:, None, None]
    field_ = trans[rows, bins[:, :, None], bins[:, None, :]]
    if image_size is not None and image_size < length:
        avg = paa_matrix(length, image_size)
        field_ = avg @ field_ @ avg.T
    return field_


def recurrence_plot(
    windows: np.ndarray,
    dimension: int = 2,
    delay: int = 1,
    threshold: float | None = None,
) -> np.ndarray:
    """Recurrence plot of each row using time-delay embedding.

    Rows are embedded as vectors ``(x_t, x_{t+delay}, ...)`` of ``dimension`` entries.
    The output is the Euclidean distance matrix between embedded states
    (``threshold=None``) or its binarised version ``distance <= threshold``.
    The image side is ``length - (dimension - 1) * delay``.
    """
    x = np.atleast_2d(np.asarray(windows, dtype=float))
    if dimension < 1 or delay < 1:
        raise ValueError("dimension and delay must be >= 1")
    side = x.shape[1] - (dimension - 1) * delay
    if side < 2:
        raise ValueError("window too short for the requested embedding")
    emb = np.stack([x[:, k * delay : k * delay + side] for k in range(dimension)], axis=2)
    sq = (emb**2).sum(axis=2)
    d2 = sq[:, :, None] + sq[:, None, :] - 2.0 * np.einsum("nik,njk->nij", emb, emb)
    dist = np.sqrt(np.maximum(d2, 0.0))
    if threshold is None:
        return dist
    return (dist <= threshold).astype(float)


@dataclass
class ImageTransformer:
    """Fit-on-normal-data wrapper that turns windows into square images.

    Parameters
    ----------
    name:
        One of ``TRANSFORM_NAMES``.
    image_size:
        Side length of the output image. Windows longer than this are reduced with PAA.
    scale:
        ``"global"`` fits value range, bin edges and distance scale on the training
        series. ``"window"`` rescales every window on its own (textbook behaviour).
    n_bins:
        Number of quantile bins of the Markov Transition Field.
    rp_dimension, rp_delay, rp_threshold:
        Recurrence plot embedding and optional binarisation threshold, expressed in
        units of the training standard deviation.
    """

    name: str
    image_size: int = 24
    scale: str = "global"
    n_bins: int = 8
    rp_dimension: int = 2
    rp_delay: int = 1
    rp_threshold: float | None = None
    lo_: float = field(default=0.0, init=False)
    hi_: float = field(default=1.0, init=False)
    std_: float = field(default=1.0, init=False)
    edges_: np.ndarray | None = field(default=None, init=False)
    fitted_: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.name not in TRANSFORM_NAMES:
            raise ValueError(f"unknown transform '{self.name}', choose from {TRANSFORM_NAMES}")
        if self.scale not in ("global", "window"):
            raise ValueError("scale must be 'global' or 'window'")

    def fit(self, train_series: np.ndarray) -> ImageTransformer:
        """Estimate scaling constants from an anomaly-free series."""
        x = np.asarray(train_series, dtype=float).ravel()
        lo, hi = np.quantile(x, [0.001, 0.999])
        margin = 0.1 * max(hi - lo, 1e-12)
        self.lo_, self.hi_ = float(lo - margin), float(hi + margin)
        self.std_ = float(max(x.std(), 1e-12))
        probs = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
        self.edges_ = np.quantile(x, probs)
        self.fitted_ = True
        return self

    def transform(self, windows: np.ndarray) -> np.ndarray:
        """Return images of shape ``(n_windows, image_size, image_size)``."""
        x = np.atleast_2d(np.asarray(windows, dtype=float))
        if self.scale == "global" and not self.fitted_:
            raise RuntimeError("call fit() before transform() when scale='global'")
        if self.name in ("gasf", "gadf"):
            reduced = paa(x, self.image_size)
            if self.scale == "global":
                scaled = scale_to_unit(reduced, self.lo_, self.hi_)
            else:
                lo = reduced.min(axis=1, keepdims=True)
                hi = reduced.max(axis=1, keepdims=True)
                scaled = scale_to_unit(reduced, lo, hi)
            kind = "summation" if self.name == "gasf" else "difference"
            return gramian_angular_field(scaled, kind)
        if self.name == "mtf":
            if self.scale == "global":
                edges = self.edges_
            else:
                probs = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
                edges = np.quantile(x, probs, axis=1).T
            return markov_transition_field(x, edges, self.image_size)
        extra = (self.rp_dimension - 1) * self.rp_delay
        reduced = paa(x, self.image_size + extra)
        if self.scale == "global":
            reduced = reduced / self.std_
        else:
            reduced = reduced / np.maximum(reduced.std(axis=1, keepdims=True), 1e-12)
        return recurrence_plot(reduced, self.rp_dimension, self.rp_delay, self.rp_threshold)
