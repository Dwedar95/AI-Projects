"""Render the pipeline results as Markdown: a results report and a model card."""

from __future__ import annotations

from typing import Any


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def models_table(results: dict[str, Any]) -> str:
    rows = []
    for name, rep in results["models"].items():
        mark = " (selected)" if name == results["best_model"] else ""
        rows.append([
            name + mark, str(rep["n_train_used"]),
            f"{rep['cv_rmse_mean']:.5f} +/- {rep['cv_rmse_std']:.5f}",
            f"{rep['test']['rmse']:.5f}", f"{rep['test']['mae']:.5f}",
            f"{rep['test']['max_abs_error']:.4f}", f"{rep['test']['r2']:.4f}",
            f"{rep['search_seconds']:.1f}", f"{rep['predict_us_per_sample']:.1f}",
        ])  # fmt: skip
    return _table(
        ["model", "train rows", "CV RMSE (5-fold)", "test RMSE", "test MAE", "test max error",
         "test R2", "search time (s)", "batched predict (us/sample)"], rows,
    )  # fmt: skip


def uncertainty_table(results: dict[str, Any]) -> str:
    rows = [
        [name, row["native_method"], f"{row['native_coverage']:.3f}",
         f"{row['native_mean_width']:.4f}", f"{row['conformal_coverage']:.3f}",
         f"{row['conformal_mean_width']:.4f}"]
        for name, row in results["uncertainty"]["models"].items()
    ]  # fmt: skip
    return _table(["model", "native interval", "native coverage", "native mean width",
                   "conformal coverage", "conformal mean width"], rows)  # fmt: skip


def importance_table(results: dict[str, Any]) -> str:
    items = sorted(results["permutation_importance"].items(), key=lambda kv: -kv[1]["mean"])
    rows = [[name, f"{v['mean']:.4f} +/- {v['std']:.4f}"] for name, v in items]
    return _table(["design variable", "drop in test R2 when shuffled"], rows)


def throughput_table(results: dict[str, Any]) -> str:
    speed = results["throughput_evals_per_s"]
    bem = speed["bem"]
    rows = [
        [name, f"{v['single_call_per_s']:.0f}",
         f"{v['single_call_per_s'] / bem['single_call_per_s']:.1f}x",
         f"{v['batched_per_s']:.0f}", f"{v['batched_per_s'] / bem['batched_per_s']:.1f}x"]
        for name, v in speed.items()
    ]  # fmt: skip
    return _table(["model", "single-row calls per s", "vs BEM", "batched rows per s", "vs BEM"],
                  rows)  # fmt: skip


def optimisation_table(results: dict[str, Any]) -> str:
    rows = []
    for airfoil, block in results["optimisation"]["airfoils"].items():
        d = block["direct_bem"]
        rows.append([airfoil, "direct BEM", f"{d['x']['tsr']:.2f}", f"{d['x']['pitch_deg']:.2f}",
                     f"{d['x']['chord_scale']:.3f}", f"{d['x']['twist_scale']:.3f}",
                     f"{d['objective_cp']:.4f}", f"{d['objective_cp']:.4f}", "0.0000",
                     str(d["n_evaluations"]), f"{d['seconds']:.1f}", "1.0x", "1.0x"])  # fmt: skip
        for name, s in block["surrogates"].items():
            rows.append([airfoil, name, f"{s['x']['tsr']:.2f}", f"{s['x']['pitch_deg']:.2f}",
                         f"{s['x']['chord_scale']:.3f}", f"{s['x']['twist_scale']:.3f}",
                         f"{s['objective_cp']:.4f}", f"{s['cp_bem_at_surrogate_optimum']:.4f}",
                         f"{s['cp_gap_to_direct']:.4f}", str(s["n_evaluations"]),
                         f"{s['seconds']:.1f}", f"{s['optimiser_speed_up']:.1f}x",
                         f"{s['speed_up_including_setup']:.2f}x"])  # fmt: skip
    return _table(
        ["airfoil", "objective model", "TSR", "pitch (deg)", "chord scale", "twist scale",
         "Cp predicted", "Cp by BEM at that point", "Cp gap to direct optimum", "evaluations",
         "optimiser time (s)", "speed-up", "speed-up incl. data and tuning"], rows,
    )  # fmt: skip


def aep_table(results: dict[str, Any]) -> str:
    rows = []
    for airfoil, block in results["power"]["airfoils"].items():
        for mean, row in block["aep_by_mean_wind_speed"].items():
            rows.append([airfoil, mean, f"{row['aep_bem_gwh']:.2f}",
                         f"{row['aep_surrogate_gwh']:.2f}",
                         f"{row['relative_error_percent']:+.3f}",
                         f"{row['capacity_factor_bem']:.3f}"])  # fmt: skip
    return _table(["airfoil", "mean wind speed (m/s)", "AEP from BEM (GWh)",
                   "AEP from surrogate (GWh)", "relative error (%)", "capacity factor (BEM)"],
                  rows)  # fmt: skip


def power_summary_table(results: dict[str, Any]) -> str:
    rows = []
    for airfoil, block in results["power"]["airfoils"].items():
        pitch = ", ".join(f"{u} m/s: {p:.1f}" for u, p in block["pitch_for_rated_deg_bem"].items())
        rows.append([airfoil, f"{block['max_cp_bem']:.4f}", f"{block['rated_speed_bem']:.1f}",
                     f"{block['rated_speed_surrogate']:.1f}",
                     f"{block['max_power_curve_error_kw']:.1f}", pitch])  # fmt: skip
    return _table(["airfoil", "max Cp on control grid (BEM)", "rated speed BEM (m/s)",
                   "rated speed surrogate (m/s)", "max power curve error (kW)",
                   "BEM pitch for rated power (deg)"], rows)  # fmt: skip


def render_results(results: dict[str, Any]) -> str:
    """Full results report."""
    meta = results["meta"]
    cal = results["uncertainty"].get("gaussian_process_calibration", {})
    curve = results["learning_curve_test_rmse"]
    parts = [
        "# Results",
        "",
        "Generated by `python -m wtsurrogate.cli pipeline`. Every number below comes from "
        "that run. The data is produced by the BEM solver in this repository with an "
        "illustrative rotor and approximate polars. It is not CFD data and not the data "
        "of the author's paper. Timings depend on the machine.",
        "",
        "## Dataset",
        "",
        f"- Latin hypercube design, seed {meta['seed']}: {meta['n_requested']} BEM runs, "
        f"{meta['n_failed_bem']} dropped for non-convergence, {meta['n_kept']} kept",
        f"- BEM wall time for the whole design: {meta['bem_seconds']:.2f} s (vectorised)",
        f"- split: {meta['split_sizes']['train']} train, "
        f"{meta['split_sizes']['calibration']} calibration, {meta['split_sizes']['test']} test",
        f"- Cp range in the dataset: {meta['cp_range'][0]:.3f} to {meta['cp_range'][1]:.3f}",
        "",
        "## Surrogate models for Cp",
        "",
        models_table(results),
        "",
        f"Model selection uses the CV RMSE only. Selected: `{results['best_model']}`. "
        f"The same family and hyperparameters fitted to the thrust coefficient Ct reach "
        f"test RMSE {results['ct_model_test']['rmse']:.5f} and "
        f"R2 {results['ct_model_test']['r2']:.4f}.",
        "",
        "Learning curve of the selected model (test RMSE by number of training rows): "
        + ", ".join(f"{n}: {v:.5f}" for n, v in curve.items()) + ".",
        "",
        "## Uncertainty (nominal coverage 0.90, measured on the test split)",
        "",
        uncertainty_table(results),
        "",
        "Gaussian process coverage at other nominal levels: "
        + ", ".join(f"{k}: {v:.3f}" for k, v in cal.items()) + ".",
        "",
        "## Permutation importance (selected model)",
        "",
        importance_table(results),
        "",
        "Largest absolute difference between the partial dependence of the surrogate and "
        "of the BEM model: "
        + ", ".join(f"{k}: {v:.4f}" for k, v in
                    results["partial_dependence_max_abs_error"].items()) + ".",
        "",
        "## Evaluation throughput",
        "",
        throughput_table(results),
        "",
        "## Surrogate-based optimisation of Cp "
        f"(wind speed {results['optimisation']['wind_speed']:g} m/s)",
        "",
        "Same differential evolution settings for every objective, one objective call per "
        "candidate. `Cp by BEM at that point` re-evaluates the surrogate optimum with the "
        "BEM model.",
        "",
        optimisation_table(results),
        "",
        "## Power curve and AEP (baseline geometry)",
        "",
        power_summary_table(results),
        "",
        aep_table(results),
        "",
    ]  # fmt: skip
    return "\n".join(parts)


def render_model_card(results: dict[str, Any]) -> str:
    """Model card for the selected Cp surrogate (after Mitchell et al., 2019)."""
    meta = results["meta"]
    best = results["best_model"]
    rep = results["models"][best]
    unc = results["uncertainty"]["models"][best]
    bounds = "\n".join(f"- `{k}`: {lo:g} to {hi:g}" for k, (lo, hi) in meta["bounds"].items())
    params = "\n".join(f"- `{k}`: `{v}`" for k, v in rep["best_params"].items())
    first = next(iter(results["optimisation"]["airfoils"].values()))["surrogates"][best]
    return "\n".join([
        "# Model card: Cp surrogate for the illustrative 10 MW-class rotor",
        "",
        "Generated by `python -m wtsurrogate.cli pipeline` from the same run as "
        "`docs/results.md`.",
        "",
        "## Model details",
        "",
        f"- Model family: `{best}` (scikit-learn), selected among "
        f"{len(results['models'])} families by 5-fold CV RMSE",
        "- Output: rotor power coefficient Cp (dimensionless)",
        f"- Inputs: {', '.join('`' + c + '`' for c in meta['raw_columns'])}, expanded to "
        f"{len(meta['feature_names'])} features by `FeatureBuilder`",
        "- Selected search-space values (for the Gaussian process the kernel is shown with "
        "its initial values; length scales and noise are then fitted by marginal likelihood):",
        params,
        f"- Training rows used: {rep['n_train_used']}",
        "",
        "## Intended use",
        "",
        "- Demonstration of a surrogate-modelling workflow: fast screening of operating "
        "points and blade scaling factors, optimisation studies, teaching",
        "- Out of scope: turbine design decisions, load calculations, certification, any "
        "use outside the sampled bounds",
        "",
        "## Training data",
        "",
        f"- {meta['n_kept']} steady BEM solutions on a Latin hypercube design (seed "
        f"{meta['seed']}), airfoil families: {', '.join(meta['airfoils'])}",
        "- The rotor geometry is illustrative and the polars are analytical "
        "approximations. The data is not CFD data and not measured data",
        "- Valid input range:",
        bounds,
        "",
        "## Evaluation",
        "",
        f"- 5-fold CV RMSE: {rep['cv_rmse_mean']:.5f} +/- {rep['cv_rmse_std']:.5f}",
        f"- Held-out test split ({meta['split_sizes']['test']} rows): RMSE "
        f"{rep['test']['rmse']:.5f}, MAE {rep['test']['mae']:.5f}, max absolute error "
        f"{rep['test']['max_abs_error']:.4f}, R2 {rep['test']['r2']:.4f}",
        f"- 90 percent interval, native ({unc['native_method']}): coverage "
        f"{unc['native_coverage']:.3f}, mean width {unc['native_mean_width']:.4f}",
        f"- 90 percent interval, split conformal: coverage {unc['conformal_coverage']:.3f}, "
        f"mean width {unc['conformal_mean_width']:.4f}",
        f"- Optimisation check: the surrogate optimum, re-evaluated with BEM, is "
        f"{first['cp_gap_to_direct']:.4f} Cp below the direct BEM optimum",
        "",
        "## Limitations and risks",
        "",
        "- The surrogate inherits every simplification of the BEM generator: steady "
        "uniform inflow, no yaw, shear, tower or aeroelastic effects, one airfoil family "
        "along the whole span, approximate stall model",
        "- It reproduces the generator, not a real turbine. Agreement with the generator "
        "says nothing about agreement with CFD or measurements",
        "- Errors are largest at the edges of the design space and in stalled operation "
        "(low tip-speed ratio, negative pitch). Extrapolation outside the bounds is not "
        "supported",
        "- The test split comes from the same Latin hypercube as the training data, so "
        "the reported error is an interpolation error",
        "- Prediction intervals were checked only for marginal coverage on one test "
        "split. They are not guaranteed to hold conditionally in every region",
        "",
    ])  # fmt: skip
