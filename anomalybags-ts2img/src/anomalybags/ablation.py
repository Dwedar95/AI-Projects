"""Ablation runner with fixed seeds on the synthetic benchmark.

Every study returns plain dictionaries of ``mean`` and ``std`` over dataset seeds so
the numbers can be written to JSON and rendered as Markdown tables. All metrics are
point-level and unadjusted unless a key starts with ``pa_``.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest

from anomalybags.detectors import DETECTOR_NAMES
from anomalybags.ensemble import AnomalyBags, EnsembleConfig, aggregate_scores
from anomalybags.metrics import evaluate, pr_auc
from anomalybags.synthetic import ANOMALY_TYPES, Benchmark, make_benchmark
from anomalybags.transforms import TRANSFORM_NAMES, paa
from anomalybags.windowing import sliding_windows, windows_to_points

SCENARIOS: dict[str, tuple[str, ...]] = {**{k: (k,) for k in ANOMALY_TYPES}, "mixed": ANOMALY_TYPES}
METRICS: tuple[str, ...] = ("roc_auc", "pr_auc", "best_f1", "pa_best_f1")


def _summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "n": int(arr.size)}


def _summarise(runs: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {m: _summary([r[m] for r in runs]) for m in METRICS}


def _benchmarks(
    scenario: str, seeds: tuple[int, ...], difficulty: float, n_events: int
) -> list[Benchmark]:
    return [
        make_benchmark(seed=s, kinds=SCENARIOS[scenario], difficulty=difficulty, n_events=n_events)
        for s in seeds
    ]


def run_ensemble(bench: Benchmark, config: EnsembleConfig) -> dict[str, float]:
    """Fit on the train split, score the test split, return point-level metrics."""
    model = AnomalyBags(config).fit(bench.train)
    return evaluate(bench.labels, model.score_points(bench.test), with_point_adjust=True)


def raw_window_baseline(
    bench: Benchmark, window: int = 64, stride: int = 4, size: int = 24, seed: int = 0
) -> dict[str, float]:
    """Non-image baseline: Isolation Forest on PAA-reduced raw windows."""
    train = paa(sliding_windows(bench.train, window, stride), size)
    test = paa(sliding_windows(bench.test, window, stride), size)
    forest = IsolationForest(n_estimators=100, random_state=seed).fit(train)
    points = windows_to_points(-forest.score_samples(test), bench.test.size, window, stride)
    return evaluate(bench.labels, points, with_point_adjust=True)


def study_transforms(
    base: EnsembleConfig, seeds: tuple[int, ...], difficulty: float, n_events: int
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Single transform vs all transforms vs a raw-window baseline, per anomaly type."""
    out: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for scenario in SCENARIOS:
        benches = _benchmarks(scenario, seeds, difficulty, n_events)
        rows: dict[str, dict[str, dict[str, float]]] = {}
        for name in TRANSFORM_NAMES:
            cfg = replace(base, transforms=(name,))
            rows[f"{name} only"] = _summarise([run_ensemble(b, cfg) for b in benches])
        rows["all transforms"] = _summarise([run_ensemble(b, base) for b in benches])
        rows["raw windows + iforest"] = _summarise(
            [raw_window_baseline(b, base.window, base.stride, base.image_size) for b in benches]
        )
        out[scenario] = rows
    return out


def study_detectors(
    base: EnsembleConfig, seeds: tuple[int, ...], difficulty: float, n_events: int
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Single detector family vs all detectors (all transforms in both cases)."""
    out: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for scenario in SCENARIOS:
        benches = _benchmarks(scenario, seeds, difficulty, n_events)
        rows = {}
        for name in DETECTOR_NAMES:
            cfg = replace(base, detectors=(name,))
            rows[f"{name} only"] = _summarise([run_ensemble(b, cfg) for b in benches])
        rows["all detectors"] = _summarise([run_ensemble(b, base) for b in benches])
        out[scenario] = rows
    return out


def study_members(
    base: EnsembleConfig, seeds: tuple[int, ...], difficulty: float, n_events: int
) -> dict[str, dict[str, float]]:
    """Compare the aggregated ensemble with its own individual members (PR-AUC).

    ``best member`` is an oracle: it is selected with the test labels and is
    therefore not available in practice. It shows how much the aggregation loses
    against perfect member selection.
    """
    out: dict[str, dict[str, float]] = {}
    for scenario in SCENARIOS:
        ens, best, median, worst, wins = [], [], [], [], []
        for bench in _benchmarks(scenario, seeds, difficulty, n_events):
            model = AnomalyBags(base).fit(bench.train)
            members = model.member_scores(bench.test)
            n = bench.test.size
            per_member = [
                pr_auc(bench.labels, windows_to_points(row, n, base.window, base.stride))
                for row in members
            ]
            agg = windows_to_points(
                aggregate_scores(members, base.aggregation), n, base.window, base.stride
            )
            score = pr_auc(bench.labels, agg)
            ens.append(score)
            best.append(max(per_member))
            median.append(float(np.median(per_member)))
            worst.append(min(per_member))
            wins.append(float(np.mean(np.asarray(per_member) > score)))
        out[scenario] = {
            "ensemble": float(np.mean(ens)),
            "best_member_oracle": float(np.mean(best)),
            "median_member": float(np.mean(median)),
            "worst_member": float(np.mean(worst)),
            "frac_members_better": float(np.mean(wins)),
        }
    return out


def study_ensemble_size(
    base: EnsembleConfig,
    seeds: tuple[int, ...],
    difficulty: float,
    n_events: int,
    sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    model_seeds: tuple[int, ...] = (0, 1, 2),
) -> dict[str, dict[str, dict[str, float]]]:
    """Effect of the number of members on the mixed scenario.

    Variation comes from dataset seeds and from ensemble seeds (which members are
    drawn), so the standard deviation shows how stable an ensemble of that size is.
    """
    benches = _benchmarks("mixed", seeds, difficulty, n_events)
    out = {}
    for size in sizes:
        runs = [
            run_ensemble(b, replace(base, n_estimators=size, seed=ms))
            for b in benches
            for ms in model_seeds
        ]
        out[str(size)] = _summarise(runs)
    return out


def study_window_length(
    base: EnsembleConfig,
    seeds: tuple[int, ...],
    difficulty: float,
    n_events: int,
    windows: tuple[int, ...] = (32, 64, 128),
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Effect of the window length, per anomaly type."""
    out: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for scenario in SCENARIOS:
        benches = _benchmarks(scenario, seeds, difficulty, n_events)
        out[scenario] = {
            str(w): _summarise([run_ensemble(b, replace(base, window=w)) for b in benches])
            for w in windows
        }
    return out


def study_scoring(
    base: EnsembleConfig, seeds: tuple[int, ...], difficulty: float, n_events: int
) -> dict[str, dict[str, dict[str, float]]]:
    """Normalisation x aggregation rule and window scaling mode on the mixed scenario."""
    benches = _benchmarks("mixed", seeds, difficulty, n_events)
    out = {}
    for norm in ("zscore", "robust", "minmax", "rank"):
        for agg in ("mean", "median", "max", "trimmed_mean"):
            cfg = replace(base, normalisation=norm, aggregation=agg)
            out[f"{norm} / {agg}"] = _summarise([run_ensemble(b, cfg) for b in benches])
    cfg = replace(base, scale="window")
    out["zscore / mean, per-window scaling"] = _summarise([run_ensemble(b, cfg) for b in benches])
    return out


def run_all(
    seeds: tuple[int, ...] = (0, 1, 2),
    difficulty: float = 0.6,
    n_events: int = 8,
    base: EnsembleConfig | None = None,
    verbose: bool = True,
) -> dict[str, object]:
    """Run every study and return one JSON-serialisable dictionary."""
    base = base or EnsembleConfig()
    args = (base, seeds, difficulty, n_events)
    results: dict[str, object] = {
        "meta": {
            "data": "synthetic (anomalybags.synthetic.make_benchmark)",
            "dataset_seeds": list(seeds),
            "difficulty": difficulty,
            "n_events": n_events,
            "window": base.window,
            "stride": base.stride,
            "image_size": base.image_size,
            "n_estimators": base.n_estimators,
            "metric_level": "point",
        }
    }
    studies = {
        "transforms": study_transforms,
        "detectors": study_detectors,
        "members": study_members,
        "ensemble_size": study_ensemble_size,
        "window_length": study_window_length,
        "scoring": study_scoring,
    }
    for name, func in studies.items():
        start = time.perf_counter()
        results[name] = func(*args)
        if verbose:
            print(f"study '{name}' finished in {time.perf_counter() - start:.1f} s", flush=True)
    return results


def _cell(stat: dict[str, float]) -> str:
    return f"{stat['mean']:.3f} +/- {stat['std']:.3f}"


def scenario_table(study: dict[str, dict[str, dict[str, dict[str, float]]]], metric: str) -> str:
    """Markdown table with one row per configuration and one column per scenario."""
    scenarios = list(study)
    rows = list(next(iter(study.values())))
    lines = ["| configuration | " + " | ".join(scenarios) + " |", "|---|" + "---|" * len(scenarios)]
    for row in rows:
        means = {s: study[s][row][metric]["mean"] for s in scenarios}
        cells = []
        for s in scenarios:
            top = max(study[s][r][metric]["mean"] for r in rows)
            text = _cell(study[s][row][metric])
            cells.append(f"**{text}**" if means[s] >= top - 1e-12 else text)
        lines.append(f"| {row} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def flat_table(study: dict[str, dict[str, dict[str, float]]], first_column: str) -> str:
    """Markdown table with one row per configuration and one column per metric."""
    lines = [f"| {first_column} | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    for row, stats in study.items():
        lines.append(f"| {row} | " + " | ".join(_cell(stats[m]) for m in METRICS) + " |")
    return "\n".join(lines)


def members_table(study: dict[str, dict[str, float]]) -> str:
    cols = ["ensemble", "best_member_oracle", "median_member", "worst_member",
            "frac_members_better"]  # fmt: skip
    lines = ["| scenario | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for scenario, row in study.items():
        lines.append(f"| {scenario} | " + " | ".join(f"{row[c]:.3f}" for c in cols) + " |")
    return "\n".join(lines)


def render_markdown(results: dict[str, object]) -> str:
    """Render all studies as a Markdown report."""
    meta = results["meta"]
    assert isinstance(meta, dict)
    parts = [
        "# Ablation results (synthetic benchmark)",
        "",
        "Generated by `python -m anomalybags.cli ablation`. All data is synthetic. "
        "Values are mean +/- standard deviation over dataset seeds "
        f"{meta['dataset_seeds']} (anomaly difficulty {meta['difficulty']}, "
        f"{meta['n_events']} events per test series, window {meta['window']}, "
        f"stride {meta['stride']}, image size {meta['image_size']}, "
        f"{meta['n_estimators']} members). Metrics are point-level and unadjusted. "
        "`pa_best_f1` uses the point-adjust protocol, which is optimistic and is "
        "shown only for comparison with papers that report it. "
        "Bold marks the best mean in a column.",
        "",
        "## 1. Single transform vs all transforms (PR-AUC)",
        "",
        scenario_table(results["transforms"], "pr_auc"),  # type: ignore[arg-type]
        "",
        "ROC-AUC for the same runs:",
        "",
        scenario_table(results["transforms"], "roc_auc"),  # type: ignore[arg-type]
        "",
        "## 2. Single detector family vs all detectors (PR-AUC)",
        "",
        scenario_table(results["detectors"], "pr_auc"),  # type: ignore[arg-type]
        "",
        "## 3. Ensemble vs its own members (PR-AUC)",
        "",
        "`best_member_oracle` is picked with test labels and is not available in "
        "practice. `frac_members_better` is the share of members whose individual "
        "PR-AUC exceeds the aggregated score.",
        "",
        members_table(results["members"]),  # type: ignore[arg-type]
        "",
        "## 4. Ensemble size (mixed scenario)",
        "",
        "Standard deviation is over dataset seeds and ensemble seeds.",
        "",
        flat_table(results["ensemble_size"], "members"),  # type: ignore[arg-type]
        "",
        "## 5. Window length (PR-AUC)",
        "",
        scenario_table(_transpose(results["window_length"]), "pr_auc"),  # type: ignore[arg-type]
        "",
        "## 6. Score normalisation and aggregation (mixed scenario)",
        "",
        flat_table(results["scoring"], "normalisation / aggregation"),  # type: ignore[arg-type]
        "",
    ]
    return "\n".join(parts)


def _transpose(study: dict[str, dict[str, dict]]) -> dict[str, dict[str, dict]]:
    """Prefix window-length row names so the table reads 'window 32', 'window 64', ..."""
    return {s: {f"window {w}": v for w, v in rows.items()} for s, rows in study.items()}


def save_results(results: dict[str, object], out_dir: str | Path) -> tuple[Path, Path]:
    """Write ``ablation_results.json`` and ``results.md`` to ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "ablation_results.json"
    md_path = out / "results.md"
    json_path.write_text(json.dumps(results, indent=2) + "\n")
    md_path.write_text(render_markdown(results))
    return json_path, md_path
