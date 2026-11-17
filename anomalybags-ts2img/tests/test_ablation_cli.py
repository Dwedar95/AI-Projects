from anomalybags.ablation import (
    flat_table,
    raw_window_baseline,
    run_ensemble,
    scenario_table,
    study_ensemble_size,
)
from anomalybags.cli import main
from anomalybags.ensemble import EnsembleConfig


def test_run_ensemble_and_baseline_return_metrics(small_bench):
    result = run_ensemble(small_bench, EnsembleConfig(n_estimators=2, transforms=("rp",)))
    baseline = raw_window_baseline(small_bench)
    for metrics in (result, baseline):
        assert 0.0 <= metrics["pr_auc"] <= 1.0
        assert metrics["pa_best_f1"] >= metrics["best_f1"]


def test_ensemble_size_study_and_table():
    base = EnsembleConfig(detectors=("pca", "iforest"), transforms=("gasf",))
    study = study_ensemble_size(base, (0,), 1.0, 6, sizes=(1, 2), model_seeds=(0, 1))
    assert set(study) == {"1", "2"}
    assert study["1"]["pr_auc"]["n"] == 2
    table = flat_table(study, "members")
    assert table.count("\n") == 3 and "+/-" in table


def test_scenario_table_marks_best_mean():
    stat = lambda m: {"pr_auc": {"mean": m, "std": 0.0}}  # noqa: E731
    study = {"spike": {"a": stat(0.2), "b": stat(0.6)}, "mixed": {"a": stat(0.9), "b": stat(0.1)}}
    lines = scenario_table(study, "pr_auc").splitlines()
    assert lines[2] == "| a | 0.200 +/- 0.000 | **0.900 +/- 0.000** |"
    assert lines[3] == "| b | **0.600 +/- 0.000** | 0.100 +/- 0.000 |"


def test_cli_demo_prints_metrics(capsys):
    assert main(["demo", "--n-estimators", "4"]) == 0
    out = capsys.readouterr().out
    assert "SYNTHETIC" in out and "PR-AUC" in out and "point-adjusted" in out
