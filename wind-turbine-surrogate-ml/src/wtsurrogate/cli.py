"""Command line interface: ``wtsurrogate demo`` and ``wtsurrogate pipeline``."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from wtsurrogate.bem import solve_bem
from wtsurrogate.pipeline import run_pipeline
from wtsurrogate.report import render_model_card, render_results
from wtsurrogate.rotor import reference_rotor


def print_summary(results: dict[str, Any]) -> None:
    """Print the headline numbers of a pipeline run."""
    best = results["best_model"]
    print("\nsummary")
    for name, rep in results["models"].items():
        print(f"  {name:<18} CV RMSE {rep['cv_rmse_mean']:.5f}  test RMSE "
              f"{rep['test']['rmse']:.5f}  test R2 {rep['test']['r2']:.4f}")  # fmt: skip
    unc = results["uncertainty"]["models"][best]
    print(f"  selected model: {best}, 90 percent interval coverage {unc['native_coverage']:.3f} "
          f"(native), {unc['conformal_coverage']:.3f} (conformal)")  # fmt: skip
    for airfoil, block in results["optimisation"]["airfoils"].items():
        direct, sur = block["direct_bem"], block["surrogates"][best]
        print(f"  optimum {airfoil}: direct BEM Cp {direct['objective_cp']:.4f} in "
              f"{direct['seconds']:.1f} s, surrogate optimum verified by BEM Cp "
              f"{sur['cp_bem_at_surrogate_optimum']:.4f} in {sur['seconds']:.1f} s "
              f"({sur['optimiser_speed_up']:.1f}x)")  # fmt: skip
    for airfoil, block in results["power"]["airfoils"].items():
        row = block["aep_by_mean_wind_speed"]["8.5"]
        print(f"  AEP {airfoil} at 8.5 m/s mean wind: BEM {row['aep_bem_gwh']:.2f} GWh, "
              f"surrogate {row['aep_surrogate_gwh']:.2f} GWh "
              f"({row['relative_error_percent']:+.3f} percent)")  # fmt: skip


def run_demo() -> dict[str, Any]:
    """Small end-to-end run: BEM check, 400-sample dataset, two surrogates."""
    rotor = reference_rotor()
    tsr = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    res = solve_bem(rotor, 9.0, tsr, 0.0)
    print("BEM check, illustrative 10 MW-class rotor (naca4415 approximation), 9 m/s, pitch 0")
    for lam, cp, ct in zip(tsr, res.cp, res.ct, strict=True):
        print(f"  TSR {lam:4.1f}  Cp {cp:.4f}  Ct {ct:.4f}")
    print("\npipeline on a reduced budget (400 BEM samples, 2 model families)")
    results, _ = run_pipeline(n_samples=400, budget=0.25, optimiser_maxiter=8,
                              models=("gaussian_process", "polynomial_ridge"))  # fmt: skip
    print_summary(results)
    return results


def run_full(out_dir: str, n_samples: int, seed: int, figures: bool) -> dict[str, Any]:
    """Full pipeline: writes results.json, results.md, model_card.md, dataset and figures."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results, artifacts = run_pipeline(n_samples=n_samples, seed=seed)
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (out / "results.md").write_text(render_results(results))
    (out / "model_card.md").write_text(render_model_card(results))
    artifacts.dataset.to_csv(out / "bem_dataset.csv")
    print(f"wrote results.json, results.md, model_card.md and bem_dataset.csv to {out}")
    if figures:
        from wtsurrogate.plots import make_all_figures

        for path in make_all_figures(artifacts, results, out):
            print(f"wrote {path}")
    print_summary(results)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wtsurrogate", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo", help="reduced-budget run that prints real numbers")
    full = sub.add_parser("pipeline", help="full workflow, writes docs")
    full.add_argument("--out", default="docs")
    full.add_argument("--n-samples", type=int, default=1500)
    full.add_argument("--seed", type=int, default=0)
    full.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "demo":
        run_demo()
    else:
        run_full(args.out, args.n_samples, args.seed, not args.no_figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
