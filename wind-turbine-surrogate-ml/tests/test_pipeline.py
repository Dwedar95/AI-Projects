import json

import pytest

from wtsurrogate.pipeline import run_pipeline, split_indices
from wtsurrogate.report import render_model_card, render_results


@pytest.fixture(scope="module")
def run():
    return run_pipeline(n_samples=260, budget=0.2, optimiser_maxiter=2, verbose=False,
                        models=("polynomial_ridge", "gaussian_process"))


def test_split_indices_are_disjoint_and_complete():
    splits = split_indices(100, seed=0)
    merged = sorted(i for part in splits.values() for i in part)
    assert merged == list(range(100))
    assert [len(splits[k]) for k in ("train", "calibration", "test")] == [70, 15, 15]


def test_pipeline_results_are_consistent(run):
    results, artifacts = run
    assert results["best_model"] == min(
        results["models"], key=lambda k: results["models"][k]["cv_rmse_mean"]
    )
    assert results["meta"]["n_kept"] + results["meta"]["n_failed_bem"] == 260
    assert results["models"][results["best_model"]]["test"]["r2"] > 0.9
    json.dumps(results)  # must be serialisable
    block = results["optimisation"]["airfoils"]["naca4415"]
    for row in block["surrogates"].values():
        assert row["cp_bem_at_surrogate_optimum"] <= block["direct_bem"]["objective_cp"] + 0.02
        assert row["cp_gap_to_direct"] == pytest.approx(
            block["direct_bem"]["objective_cp"] - row["cp_bem_at_surrogate_optimum"]
        )
    assert set(artifacts.pdp) == {"wind_speed", "tsr", "pitch_deg", "chord_scale", "twist_scale"}


def test_pipeline_power_and_uncertainty_sections(run):
    results, _ = run
    for block in results["power"]["airfoils"].values():
        for row in block["aep_by_mean_wind_speed"].values():
            assert 20.0 < row["aep_bem_gwh"] < 70.0
            assert abs(row["relative_error_percent"]) < 5.0
    for row in results["uncertainty"]["models"].values():
        assert 0.0 <= row["native_coverage"] <= 1.0
        assert row["conformal_mean_width"] > 0


def test_reports_render_without_dashes(run):
    results, _ = run
    text = render_results(results) + render_model_card(results)
    assert "## Surrogate models for Cp" in text and "# Model card" in text
    assert chr(0x2014) not in text and chr(0x2013) not in text
    assert results["best_model"] in text
