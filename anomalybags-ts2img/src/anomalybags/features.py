"""Image feature extractors that need no deep learning framework.

Three families, all vectorised over a batch of square images ``(n, s, s)``:

* ``stats``: intensity statistics, diagonal statistics, gradient energy and
  grey-level co-occurrence (GLCM) texture descriptors
* ``hog``: a HOG-like descriptor (magnitude-weighted orientation histograms per cell)
* ``pca``: projection of the flattened image on principal components learned from
  the training images
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FEATURE_KINDS: tuple[str, ...] = ("stats", "hog", "pca")


def intensity_statistics(images: np.ndarray) -> np.ndarray:
    """Global, diagonal and gradient statistics, 16 features per image."""
    n = images.shape[0]
    flat = images.reshape(n, -1)
    mean = flat.mean(axis=1)
    std = flat.std(axis=1)
    centred = flat - mean[:, None]
    safe = np.maximum(std, 1e-12)
    skew = (centred**3).mean(axis=1) / safe**3
    kurt = (centred**4).mean(axis=1) / safe**4 - 3.0
    q10, q50, q90 = np.quantile(flat, [0.1, 0.5, 0.9], axis=1)
    diag = np.diagonal(images, axis1=1, axis2=2)
    off = np.diagonal(images, offset=max(1, images.shape[1] // 4), axis1=1, axis2=2)
    anti = np.diagonal(images[:, :, ::-1], axis1=1, axis2=2)
    gx = np.diff(images, axis=2)
    gy = np.diff(images, axis=1)
    return np.column_stack(
        [
            mean, std, skew, kurt, q10, q50, q90, flat.min(axis=1), flat.max(axis=1),
            diag.mean(axis=1), diag.std(axis=1), off.mean(axis=1), anti.mean(axis=1),
            np.abs(gx).mean(axis=(1, 2)), np.abs(gy).mean(axis=(1, 2)),
            np.sqrt((gx**2).mean(axis=(1, 2)) + (gy**2).mean(axis=(1, 2))),
        ]
    )  # fmt: skip


def glcm_features(images: np.ndarray, vmin: float, vmax: float, levels: int = 8) -> np.ndarray:
    """Grey-level co-occurrence descriptors for horizontal and vertical neighbours.

    Images are quantised to ``levels`` grey levels on the fixed range ``[vmin, vmax]``.
    For each of the two offsets a symmetric, normalised co-occurrence matrix is built
    and summarised by contrast, homogeneity, energy, entropy and correlation
    (Haralick et al., 1973), giving 10 features per image.
    """
    n = images.shape[0]
    span = max(vmax - vmin, 1e-12)
    q = np.clip(((images - vmin) / span * levels).astype(int), 0, levels - 1)
    i_idx, j_idx = np.meshgrid(np.arange(levels), np.arange(levels), indexing="ij")
    out = []
    base = (np.arange(n) * levels * levels)[:, None]
    for a, b in ((q[:, :, :-1], q[:, :, 1:]), (q[:, :-1, :], q[:, 1:, :])):
        code = (a * levels + b).reshape(n, -1) + base
        counts = np.bincount(code.ravel(), minlength=n * levels * levels).astype(float)
        p = counts.reshape(n, levels, levels)
        p = p + p.transpose(0, 2, 1)
        p /= np.maximum(p.sum(axis=(1, 2), keepdims=True), 1e-12)
        contrast = (p * (i_idx - j_idx) ** 2).sum(axis=(1, 2))
        homogeneity = (p / (1.0 + np.abs(i_idx - j_idx))).sum(axis=(1, 2))
        energy = (p**2).sum(axis=(1, 2))
        entropy = -(p * np.log(p + 1e-12)).sum(axis=(1, 2))
        mu = (p * i_idx).sum(axis=(1, 2))
        var = (p * (i_idx - mu[:, None, None]) ** 2).sum(axis=(1, 2))
        cov = (p * (i_idx - mu[:, None, None]) * (j_idx - mu[:, None, None])).sum(axis=(1, 2))
        correlation = np.where(var > 1e-12, cov / np.maximum(var, 1e-12), 1.0)
        out.extend([contrast, homogeneity, energy, entropy, correlation])
    return np.column_stack(out)


def hog_features(images: np.ndarray, cells: int = 3, bins: int = 6) -> np.ndarray:
    """HOG-like descriptor: ``cells x cells`` orientation histograms with ``bins`` bins.

    Gradients come from central differences, orientations are unsigned (0 to pi) and
    each pixel votes with its gradient magnitude. The descriptor is L2-normalised per
    image, so it encodes the shape of the texture; magnitude is carried by ``stats``.
    """
    n, side, _ = images.shape
    if cells < 1 or cells > side:
        raise ValueError("cells must be in [1, image side]")
    gy, gx = np.gradient(images, axis=(1, 2))
    mag = np.hypot(gx, gy)
    ang = np.mod(np.arctan2(gy, gx), np.pi)
    bin_idx = np.minimum((ang / np.pi * bins).astype(int), bins - 1)
    cell_of = (np.arange(side) * cells) // side
    cell_id = cell_of[:, None] * cells + cell_of[None, :]
    width = cells * cells * bins
    code = cell_id[None, :, :] * bins + bin_idx + (np.arange(n) * width)[:, None, None]
    hist = np.bincount(code.ravel(), weights=mag.ravel(), minlength=n * width).reshape(n, width)
    return hist / np.maximum(np.linalg.norm(hist, axis=1, keepdims=True), 1e-12)


@dataclass
class PCAProjector:
    """Principal component projection of flattened images, fitted with an SVD."""

    n_components: int = 8
    mean_: np.ndarray | None = field(default=None, init=False)
    components_: np.ndarray | None = field(default=None, init=False)
    explained_variance_ratio_: np.ndarray | None = field(default=None, init=False)

    def fit(self, images: np.ndarray) -> PCAProjector:
        flat = images.reshape(images.shape[0], -1)
        self.mean_ = flat.mean(axis=0)
        _, sing, vt = np.linalg.svd(flat - self.mean_, full_matrices=False)
        k = min(self.n_components, vt.shape[0])
        self.components_ = vt[:k]
        var = sing**2
        self.explained_variance_ratio_ = var[:k] / max(var.sum(), 1e-12)
        return self

    def transform(self, images: np.ndarray) -> np.ndarray:
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("PCAProjector is not fitted")
        flat = images.reshape(images.shape[0], -1)
        return (flat - self.mean_) @ self.components_.T


@dataclass
class FeatureExtractor:
    """Concatenate feature families and standardise them with training statistics."""

    kinds: tuple[str, ...] = FEATURE_KINDS
    pca_components: int = 8
    hog_cells: int = 3
    hog_bins: int = 6
    glcm_levels: int = 8
    vmin_: float = field(default=0.0, init=False)
    vmax_: float = field(default=1.0, init=False)
    pca_: PCAProjector | None = field(default=None, init=False)
    mean_: np.ndarray | None = field(default=None, init=False)
    std_: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        unknown = set(self.kinds) - set(FEATURE_KINDS)
        if unknown or not self.kinds:
            raise ValueError(f"feature kinds must be a non-empty subset of {FEATURE_KINDS}")

    def _raw(self, images: np.ndarray) -> np.ndarray:
        blocks = []
        if "stats" in self.kinds:
            blocks.append(intensity_statistics(images))
            blocks.append(glcm_features(images, self.vmin_, self.vmax_, self.glcm_levels))
        if "hog" in self.kinds:
            blocks.append(hog_features(images, self.hog_cells, self.hog_bins))
        if "pca" in self.kinds:
            assert self.pca_ is not None
            blocks.append(self.pca_.transform(images))
        return np.column_stack(blocks)

    def fit(self, images: np.ndarray) -> FeatureExtractor:
        """Learn the grey-level range, PCA basis and standardisation from normal images."""
        images = np.asarray(images, dtype=float)
        self.vmin_, self.vmax_ = (float(v) for v in np.quantile(images, [0.01, 0.99]))
        if "pca" in self.kinds:
            self.pca_ = PCAProjector(self.pca_components).fit(images)
        raw = self._raw(images)
        self.mean_ = raw.mean(axis=0)
        self.std_ = np.maximum(raw.std(axis=0), 1e-8)
        return self

    def transform(self, images: np.ndarray) -> np.ndarray:
        """Standardised feature matrix ``(n_images, n_features)``."""
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("FeatureExtractor is not fitted")
        raw = self._raw(np.asarray(images, dtype=float))
        return np.clip((raw - self.mean_) / self.std_, -50.0, 50.0)

    def fit_transform(self, images: np.ndarray) -> np.ndarray:
        return self.fit(images).transform(images)
