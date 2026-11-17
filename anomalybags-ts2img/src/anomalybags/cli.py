"""Command line interface: ``anomalybags demo`` and ``anomalybags ablation``."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

from anomalybags.ablation import raw_window_baseline, run_all, save_results
from anomalybags.ensemble import AnomalyBags, EnsembleConfig
from anomalybags.metrics import evaluate
from anomalybags.synthetic import make_benchmark


def run_demo(seed: int = 0, difficulty: float = 0.6, n_estimators: int = 16) -> dict[str, float]:
    """Fit one ensemble on a mixed synthetic benchmark and print its metrics."""
    bench = make_benchmark(seed=seed, difficulty=difficulty, n_events=8)
    print("AnomalyBags demo on SYNTHETIC data")
    print(f"train points: {bench.train.size}, test points: {bench.test.size}, "
          f"anomalous fraction: {bench.contamination:.3f}")  # fmt: skip
    for event in bench.events:
        print(f"  event {event.kind:<17} [{event.start:>4}, {event.end:>4})  "
              f"magnitude {event.magnitude:+.2f}")  # fmt: skip
    config = EnsembleConfig(n_estimators=n_estimators, seed=seed)
    start = time.perf_counter()
    model = AnomalyBags(config).fit(bench.train)
    fit_s = time.perf_counter() - start
    start = time.perf_counter()
    scores = model.score_points(bench.test)
    score_s = time.perf_counter() - start
    metrics = evaluate(bench.labels, scores, with_point_adjust=True)
    baseline = raw_window_baseline(bench, config.window, config.stride, config.image_size)
    print(f"members: {len(model.members_)} (window {config.window}, stride {config.stride}, "
          f"image {config.image_size}x{config.image_size})")  # fmt: skip
    for row in model.describe()[:4]:
        print(f"  member {row['transform']:<5} + {row['detector']:<8} features={row['n_features']} "
              f"in-bag={row['in_bag']} out-of-bag={row['out_of_bag']}")  # fmt: skip
    print(f"fit time: {fit_s:.2f} s, scoring time: {score_s:.2f} s")
    print("point-level metrics (unadjusted):")
    print(f"  ensemble           ROC-AUC {metrics['roc_auc']:.3f}  PR-AUC {metrics['pr_auc']:.3f}  "
          f"best-F1 {metrics['best_f1']:.3f}")  # fmt: skip
    print(f"  raw-window iforest ROC-AUC {baseline['roc_auc']:.3f}  "
          f"PR-AUC {baseline['pr_auc']:.3f}  best-F1 {baseline['best_f1']:.3f}")  # fmt: skip
    print(f"point-adjusted best-F1 (optimistic protocol, for reference only): "
          f"{metrics['pa_best_f1']:.3f}")  # fmt: skip
    return metrics


def run_ablation(out_dir: str, seeds: Sequence[int], difficulty: float, figures: bool) -> None:
    """Run all ablation studies, write JSON and Markdown, optionally draw figures."""
    results = run_all(seeds=tuple(seeds), difficulty=difficulty)
    json_path, md_path = save_results(results, out_dir)
    print(f"wrote {json_path} and {md_path}")
    if figures:
        from anomalybags.plots import make_all_figures

        bench = make_benchmark(seed=seeds[0], difficulty=difficulty, n_events=8)
        model = AnomalyBags(EnsembleConfig(seed=seeds[0])).fit(bench.train)
        for path in make_all_figures(bench, model, results, out_dir):
            print(f"wrote {path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="anomalybags", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="fit one ensemble on synthetic data and print metrics")
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--difficulty", type=float, default=0.6)
    demo.add_argument("--n-estimators", type=int, default=16)
    abl = sub.add_parser("ablation", help="run all ablation studies")
    abl.add_argument("--out", default="docs")
    abl.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    abl.add_argument("--difficulty", type=float, default=0.6)
    abl.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "demo":
        run_demo(args.seed, args.difficulty, args.n_estimators)
    else:
        run_ablation(args.out, args.seeds, args.difficulty, not args.no_figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
