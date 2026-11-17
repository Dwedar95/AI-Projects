"""Matplotlib figures for the docs folder (Agg backend, no display needed)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from anomalybags.ensemble import AnomalyBags  # noqa: E402
from anomalybags.metrics import anomaly_segments  # noqa: E402
from anomalybags.synthetic import Benchmark  # noqa: E402
from anomalybags.transforms import TRANSFORM_NAMES, ImageTransformer  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
ANOMALY_BAND = "#d03b3b"
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", ["#fcfcfb", "#86b6ef", "#0d366b"])


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_transform_gallery(bench: Benchmark, path: str | Path, window: int = 64) -> Path:
    """One normal window and one window per anomaly type, encoded by every transform."""
    picks: list[tuple[str, int]] = [("normal", 200)]
    seen: set[str] = set()
    for event in bench.events:
        if event.kind not in seen:
            seen.add(event.kind)
            centre = (event.start + event.end) // 2
            picks.append((event.kind, max(0, min(centre - window // 2, bench.test.size - window))))
    fig, axes = plt.subplots(
        len(TRANSFORM_NAMES) + 1, len(picks), figsize=(2.2 * len(picks), 2.1 * 5),
        gridspec_kw={"height_ratios": [0.7, 1, 1, 1, 1]},
    )  # fmt: skip
    for col, (label, start) in enumerate(picks):
        source = bench.train if label == "normal" else bench.test
        segment = source[start : start + window]
        ax = axes[0, col]
        ax.plot(segment, color=SERIES[0], linewidth=1.5)
        ax.set_title(label.replace("_", " "), fontsize=9, color=INK)
        _style(ax)
        ax.set_xticks([])
        for row, name in enumerate(TRANSFORM_NAMES, start=1):
            image = ImageTransformer(name, image_size=32).fit(bench.train).transform(segment)[0]
            axes[row, col].imshow(image, cmap=SEQUENTIAL, origin="lower")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if col == 0:
                axes[row, col].set_ylabel(name.upper(), fontsize=9, color=INK)
    fig.suptitle("Synthetic windows and their image encodings", fontsize=11, color=INK, y=0.995)
    fig.tight_layout()
    return _save(fig, path)


def plot_score_timeline(
    bench: Benchmark, scores: np.ndarray, path: str | Path, threshold: float | None = None
) -> Path:
    """Test series with true anomaly segments and the ensemble score below it."""
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 4.6), sharex=True)
    top.plot(bench.test, color=SERIES[0], linewidth=0.8)
    top.set_ylabel("value (synthetic)", fontsize=8, color=MUTED)
    bottom.plot(scores, color=SERIES[1], linewidth=1.2)
    bottom.set_ylabel("ensemble score", fontsize=8, color=MUTED)
    bottom.set_xlabel("time step", fontsize=8, color=MUTED)
    if threshold is not None:
        bottom.axhline(threshold, color=MUTED, linewidth=0.8, linestyle="--")
        bottom.text(len(scores) * 0.995, threshold, "best-F1 threshold (oracle)", fontsize=7,
                    color=MUTED, ha="right", va="bottom")  # fmt: skip
    for ax in (top, bottom):
        _style(ax)
        for start, end in anomaly_segments(bench.labels):
            ax.axvspan(start - 0.5, end - 0.5, color=ANOMALY_BAND, alpha=0.16, linewidth=0)
    for event in bench.events:
        top.text((event.start + event.end) / 2, top.get_ylim()[1], event.kind.replace("_", " "),
                 fontsize=6.5, color="#52514e", ha="center", va="bottom", rotation=30)  # fmt: skip
    top.set_title("Test series, shaded true anomaly segments, and AnomalyBags score",
                  fontsize=10, color=INK, pad=44)  # fmt: skip
    fig.tight_layout()
    return _save(fig, path)


def plot_scenario_bars(
    study: dict[str, dict[str, dict[str, dict[str, float]]]],
    path: str | Path,
    metric: str = "pr_auc",
    title: str = "",
) -> Path:
    """Grouped bars: one group per scenario, one bar per configuration, std as whiskers."""
    scenarios = list(study)
    rows = list(next(iter(study.values())))
    fig, ax = plt.subplots(figsize=(11, 4.2))
    width = 0.8 / len(rows)
    for i, row in enumerate(rows):
        means = [study[s][row][metric]["mean"] for s in scenarios]
        stds = [study[s][row][metric]["std"] for s in scenarios]
        x = np.arange(len(scenarios)) + (i - (len(rows) - 1) / 2) * width
        ax.bar(x, means, width * 0.88, color=SERIES[i % len(SERIES)], label=row,
               yerr=stds, error_kw={"ecolor": "#52514e", "elinewidth": 0.8, "capsize": 2})
    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_xticklabels([s.replace("_", " ") for s in scenarios], fontsize=9, color=INK)
    ax.set_ylabel(f"{metric} (point-level, mean over seeds)", fontsize=8, color=MUTED)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    _style(ax)
    return _save(fig, path)


def plot_ensemble_size(study: dict[str, dict[str, dict[str, float]]], path: str | Path) -> Path:
    """PR-AUC and ROC-AUC against the number of members with a one-std band."""
    sizes = [int(k) for k in study]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for i, metric in enumerate(("pr_auc", "roc_auc")):
        mean = np.array([study[str(s)][metric]["mean"] for s in sizes])
        std = np.array([study[str(s)][metric]["std"] for s in sizes])
        ax.fill_between(sizes, mean - std, mean + std, color=SERIES[i], alpha=0.15, linewidth=0)
        ax.plot(sizes, mean, color=SERIES[i], linewidth=2, marker="o", markersize=5, label=metric)
        ax.text(sizes[-1] * 1.06, mean[-1], metric, fontsize=8, color="#52514e", va="center")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlim(sizes[0] * 0.9, sizes[-1] * 1.5)
    ax.set_xlabel("number of ensemble members", fontsize=8, color=MUTED)
    ax.set_ylabel("score (mixed scenario)", fontsize=8, color=MUTED)
    ax.set_title("Ensemble size: mean and one standard deviation", fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    _style(ax)
    return _save(fig, path)


def make_all_figures(
    bench: Benchmark, model: AnomalyBags, results: dict, out_dir: str | Path
) -> list[Path]:
    """Create every documentation figure and return the written paths."""
    from anomalybags.metrics import best_f1

    out = Path(out_dir)
    scores = model.score_points(bench.test)
    _, threshold = best_f1(bench.labels, scores)
    return [
        plot_transform_gallery(bench, out / "transform_gallery.png"),
        plot_score_timeline(bench, scores, out / "score_timeline.png", threshold),
        plot_scenario_bars(results["transforms"], out / "ablation_transforms.png",
                           title="Single transform vs all transforms vs raw-window baseline"),
        plot_scenario_bars(results["detectors"], out / "ablation_detectors.png",
                           title="Single detector family vs all detectors"),
        plot_ensemble_size(results["ensemble_size"], out / "ablation_ensemble_size.png"),
    ]  # fmt: skip
