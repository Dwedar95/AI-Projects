"""Matplotlib figures for the docs folder (Agg backend, no display needed)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from wtsurrogate.airfoils import AIRFOILS  # noqa: E402
from wtsurrogate.bem import solve_bem  # noqa: E402
from wtsurrogate.pipeline import PipelineArtifacts  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)


def _style(ax: plt.Axes, grid: str = "y") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis=grid, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set(color=SECONDARY, size=8)
    ax.yaxis.label.set(color=SECONDARY, size=8)
    ax.title.set(color=INK, size=10)


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_polars(path: Path) -> Path:
    """Lift and drag polars of both airfoil families at the reference Reynolds number."""
    alpha = np.radians(np.linspace(-15.0, 30.0, 451))
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for color, polar in zip(SERIES, AIRFOILS.values(), strict=False):
        cl, cd = polar.coefficients(alpha)
        left.plot(np.degrees(alpha), cl, color=color, linewidth=2, label=polar.name)
        right.plot(np.degrees(alpha), cd, color=color, linewidth=2, label=polar.name)
    left.set(xlabel="angle of attack (deg)", ylabel="lift coefficient",
             title="Approximate lift polars (thin airfoil + stall blend)")  # fmt: skip
    right.set(xlabel="angle of attack (deg)", ylabel="drag coefficient",
              title="Approximate drag polars")  # fmt: skip
    for ax in (left, right):
        _style(ax)
        ax.legend(fontsize=8, frameon=False)
    return _save(fig, path)


def plot_rotor(art: PipelineArtifacts, path: Path) -> Path:
    """Chord and twist distributions of the illustrative reference rotors."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for color, rotor in zip(SERIES, art.oracle.rotors, strict=False):
        left.plot(rotor.r, rotor.chord, color=color, linewidth=2, label=rotor.airfoil.name)
        right.plot(rotor.r, np.degrees(rotor.twist), color=color, linewidth=2,
                   label=rotor.airfoil.name)  # fmt: skip
    left.set(xlabel="radius (m)", ylabel="chord (m)", title="Illustrative chord distribution")
    right.set(xlabel="radius (m)", ylabel="twist (deg)", title="Illustrative twist distribution")
    for ax in (left, right):
        _style(ax)
        ax.legend(fontsize=8, frameon=False)
    return _save(fig, path)


def plot_cp_surface(art: PipelineArtifacts, path: Path) -> Path:
    """Cp over TSR and pitch: BEM, selected surrogate and their difference."""
    tsr = np.linspace(4.0, 12.0, 49)
    pitch = np.linspace(-3.0, 8.0, 45)
    tt, pp = np.meshgrid(tsr, pitch)
    rows = np.column_stack([np.full(tt.size, 9.0), tt.ravel(), pp.ravel(), np.ones(tt.size),
                            np.ones(tt.size), np.zeros(tt.size)])  # fmt: skip
    bem = art.oracle.cp(rows).reshape(tt.shape)
    sur = art.surrogates[art.best_name].predict(rows).reshape(tt.shape)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    levels = np.linspace(0.0, 0.5, 11)
    panels = ((axes[0], bem, "BEM"), (axes[1], sur, f"surrogate ({art.best_name})"))
    for ax, values, title in panels:
        filled = ax.contourf(tt, pp, np.clip(values, 0.0, 0.5), levels=levels, cmap=SEQUENTIAL)
        ax.set(xlabel="tip-speed ratio", ylabel="pitch (deg)", title=f"Cp, {title}, 9 m/s")
        fig.colorbar(filled, ax=ax, pad=0.02).ax.tick_params(labelsize=7, colors=MUTED)
        _style(ax, grid="both")
    err = np.abs(sur - bem)
    filled = axes[2].contourf(tt, pp, err, levels=8, cmap=SEQUENTIAL)
    axes[2].set(xlabel="tip-speed ratio", ylabel="pitch (deg)",
                title=f"|surrogate - BEM|, max {err.max():.4f}")  # fmt: skip
    fig.colorbar(filled, ax=axes[2], pad=0.02).ax.tick_params(labelsize=7, colors=MUTED)
    _style(axes[2], grid="both")
    fig.tight_layout()
    return _save(fig, path)


def plot_parity(art: PipelineArtifacts, path: Path) -> Path:
    """Predicted vs BEM Cp on the test split, one panel per model family."""
    y = art.dataset.cp[art.splits["test"]]
    names = list(art.reports)
    fig, axes = plt.subplots(1, len(names), figsize=(3.3 * len(names), 3.4), sharey=True)
    lim = (min(y.min(), -0.2), 0.52)
    for ax, name in zip(np.atleast_1d(axes), names, strict=True):
        ax.plot(lim, lim, color="#c3c2b7", linewidth=1)
        ax.scatter(y, art.predictions[name], s=9, color=SERIES[0], alpha=0.6, linewidths=0)
        rmse = art.reports[name].test["rmse"]
        ax.set(xlabel="Cp from BEM", title=f"{name}\ntest RMSE {rmse:.4f}", xlim=lim, ylim=lim)
        _style(ax, grid="both")
    np.atleast_1d(axes)[0].set_ylabel("Cp predicted")
    fig.tight_layout()
    return _save(fig, path)


def plot_importance(results: dict[str, Any], path: Path) -> Path:
    """Permutation importance of the raw design variables."""
    items = sorted(results["permutation_importance"].items(), key=lambda kv: kv[1]["mean"])
    names = [k for k, _ in items]
    mean = [v["mean"] for _, v in items]
    std = [v["std"] for _, v in items]
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.barh(names, mean, xerr=std, color=SERIES[0], height=0.6,
            error_kw={"ecolor": SECONDARY, "elinewidth": 0.8, "capsize": 2})  # fmt: skip
    for i, value in enumerate(mean):
        ax.text(value + max(mean) * 0.02 + std[i], i, f"{value:.3f}", va="center", fontsize=8,
                color=SECONDARY)  # fmt: skip
    ax.set(xlabel="drop in test R2 when the variable is shuffled",
           title=f"Permutation importance ({results['best_model']})")  # fmt: skip
    _style(ax, grid="x")
    return _save(fig, path)


def plot_partial_dependence(art: PipelineArtifacts, path: Path) -> Path:
    """Partial dependence of Cp on each design variable: surrogate vs BEM."""
    fig, axes = plt.subplots(1, len(art.pdp), figsize=(3.0 * len(art.pdp), 3.1), sharey=True)
    for ax, (variable, data) in zip(axes, art.pdp.items(), strict=True):
        ax.plot(data["grid"], data["bem"], color=SERIES[0], linewidth=2, label="BEM")
        ax.plot(data["grid"], data["surrogate"], color=SERIES[1], linewidth=2, linestyle="--",
                label="surrogate")  # fmt: skip
        ax.set(xlabel=variable, title=variable)
        _style(ax)
    axes[0].set_ylabel("mean Cp over the test rows")
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, path)


def plot_intervals(art: PipelineArtifacts, results: dict[str, Any], path: Path) -> Path:
    """Sorted test predictions of the selected model with its native 90 percent interval."""
    name = art.best_name
    y = art.dataset.cp[art.splits["test"]]
    order = np.argsort(y)
    lo, hi = art.intervals[name]
    row = results["uncertainty"]["models"][name]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    x = np.arange(len(y))
    ax.fill_between(x, lo[order], hi[order], color=SERIES[0], alpha=0.25, linewidth=0,
                    label="native 90 percent interval")  # fmt: skip
    ax.plot(x, art.predictions[name][order], color=SERIES[0], linewidth=1.5, label="prediction")
    ax.scatter(x, y[order], s=7, color=SERIES[1], linewidths=0, label="BEM", zorder=3)
    ax.set(xlabel="test rows sorted by BEM Cp", ylabel="Cp",
           title=f"{name}: measured coverage {row['native_coverage']:.3f} at nominal 0.90")
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    return _save(fig, path)


def plot_power_curve(art: PipelineArtifacts, path: Path) -> Path:
    """Power curves from BEM and from the surrogate for both airfoil families."""
    airfoils = sorted({key.split("/")[0] for key in art.power_curves})
    fig, axes = plt.subplots(1, len(airfoils), figsize=(5.2 * len(airfoils), 3.5), sharey=True)
    for ax, airfoil in zip(np.atleast_1d(axes), airfoils, strict=True):
        bem = art.power_curves[f"{airfoil}/bem"]
        sur = art.power_curves[f"{airfoil}/surrogate"]
        ax.plot(bem.speed, bem.power / 1e6, color=SERIES[0], linewidth=2, label="BEM")
        ax.plot(sur.speed, sur.power / 1e6, color=SERIES[1], linewidth=2, linestyle="--",
                label="surrogate")  # fmt: skip
        ax.set(xlabel="wind speed (m/s)", title=f"Power curve, {airfoil} rotor")
        ax.legend(fontsize=8, frameon=False, loc="lower right")
        _style(ax)
    np.atleast_1d(axes)[0].set_ylabel("electrical power (MW)")
    fig.tight_layout()
    return _save(fig, path)


def plot_bem_spanwise(art: PipelineArtifacts, path: Path) -> Path:
    """Spanwise axial induction and angle of attack at three tip-speed ratios."""
    rotor = art.oracle.rotors[0]
    tsr = np.array([5.0, 7.5, 10.0])
    res = solve_bem(rotor, 9.0, tsr, 0.0)
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for i, color in enumerate(SERIES[:3]):
        label = f"TSR {tsr[i]:g}"
        left.plot(rotor.r, res.axial_induction[i], color=color, linewidth=2, label=label)
        right.plot(rotor.r, np.degrees(res.alpha[i]), color=color, linewidth=2, label=label)
    left.set(xlabel="radius (m)", ylabel="axial induction factor",
             title="BEM solution along the span (9 m/s, pitch 0)")  # fmt: skip
    right.set(xlabel="radius (m)", ylabel="angle of attack (deg)", title="Angle of attack")
    for ax in (left, right):
        _style(ax)
        ax.legend(fontsize=8, frameon=False)
    return _save(fig, path)


def make_all_figures(art: PipelineArtifacts, results: dict[str, Any], out_dir: str | Path) -> list:
    """Create every documentation figure and return the written paths."""
    out = Path(out_dir)
    return [
        plot_polars(out / "polars.png"),
        plot_rotor(art, out / "rotor_geometry.png"),
        plot_bem_spanwise(art, out / "bem_spanwise.png"),
        plot_cp_surface(art, out / "cp_surface.png"),
        plot_parity(art, out / "parity.png"),
        plot_importance(results, out / "permutation_importance.png"),
        plot_partial_dependence(art, out / "partial_dependence.png"),
        plot_intervals(art, results, out / "uncertainty_intervals.png"),
        plot_power_curve(art, out / "power_curve.png"),
    ]
