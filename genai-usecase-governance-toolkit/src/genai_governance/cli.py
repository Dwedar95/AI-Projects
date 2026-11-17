"""Command line interface: ``genai-gov <command>``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .cards import system_card
from .fairness import fairness_report, load_predictions, performance_metrics
from .models import Portfolio, load_portfolio
from .reports import build_reports, triage_portfolio
from .risk_triage import ALL_QUESTION_IDS, DISCLAIMER, TriageResult, triage
from .roi import business_case, monte_carlo, tornado
from .scoring import build_buy_partner, score_portfolio, sensitivity_analysis


def table(headers: list[str], rows: list[list[object]]) -> str:
    """Format a plain-text table with left-aligned first column and right-aligned numbers."""
    cells = [[str(c) for c in row] for row in rows]
    widths = [max(len(h), *(len(r[i]) for r in cells)) if cells else len(h) for i, h in enumerate(headers)]
    def fmt(row: list[str]) -> str:
        return "  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(row, widths, strict=True)))
    return "\n".join([fmt(headers), "  ".join("-" * w for w in widths)] + [fmt(r) for r in cells])


def _header(portfolio: Portfolio) -> str:
    note = " (fictional example data)" if portfolio.fictional else ""
    return f"Portfolio: {portfolio.name}{note}"


def cmd_prioritise(args: argparse.Namespace) -> int:
    p = load_portfolio(args.portfolio)
    print(_header(p))
    w = p.scoring.dimension_weights
    print(f"Weights: value={w['value']:.2f} feasibility={w['feasibility']:.2f} risk={w['risk']:.2f}\n")
    rows = [[s.rank, s.id, s.name, f"{s.value:.2f}", f"{s.feasibility:.2f}", f"{s.risk:.2f}", f"{s.overall:.1f}",
             s.quadrant] for s in score_portfolio(p)]
    print(table(["#", "id", "use case", "value", "feas.", "risk", "score", "quadrant"], rows))
    return 0


def cmd_sensitivity(args: argparse.Namespace) -> int:
    p = load_portfolio(args.portfolio)
    rep = sensitivity_analysis(p, n_samples=args.samples, top_k=args.top_k, seed=args.seed,
                               concentration=args.concentration)
    print(_header(p))
    print("\nOne-at-a-time weight shifts (others rescaled):")
    rows = [[o.dimension, f"{o.delta:+.2f}", f"{o.kendall_tau:.3f}", o.max_rank_shift,
             "yes" if o.top_changed else "no", " > ".join(o.ranking[:3])] for o in rep.one_at_a_time]
    print(table(["dimension", "delta", "kendall tau", "max shift", "top changed", "top 3"], rows))
    print(f"\nMonte Carlo: {rep.n_samples} Dirichlet weight samples, concentration={rep.concentration:g}, seed={rep.seed}")
    rows = [[s.baseline_rank, s.id, f"{s.mean_rank:.2f}", s.best_rank, s.worst_rank, f"{s.p_top_k:.1%}",
             f"{s.p_same_rank:.1%}"] for s in rep.stability]
    print(table(["#", "id", "mean rank", "best", "worst", f"P(top {rep.top_k})", "P(same rank)"], rows))
    print(f"\nMean Kendall tau vs baseline: {rep.mean_kendall_tau:.3f}")
    print(f"P(rank 1 unchanged): {rep.p_top1_unchanged:.1%}")
    print("Smallest single-weight shift that changes rank 1:")
    for sp in rep.switch_points:
        if sp["delta"] is None:
            print(f"  {sp['dimension']}: no shift changes rank 1")
        else:
            print(f"  {sp['dimension']}: {sp['delta']:+.3f} (new rank 1: {sp['new_top']})")
    return 0


def cmd_build_buy(args: argparse.Namespace) -> int:
    p = load_portfolio(args.portfolio)
    print(_header(p) + "\n")
    rows = []
    for uc in p.use_cases:
        if uc.build_buy is None:
            continue
        d = build_buy_partner(uc.build_buy)
        rows.append([uc.id, d.recommendation, f"{d.scores['build']:+.2f}", f"{d.scores['buy']:+.2f}",
                     f"{d.scores['partner']:+.2f}", f"{d.margin:.2f}"])
        if args.use_case == uc.id:
            print(f"{uc.id}: {d.recommendation}")
            for r in d.rationale:
                print(f"  - {r}")
            print()
    print(table(["id", "recommendation", "build", "buy", "partner", "margin"], rows))
    return 0


def cmd_roi(args: argparse.Namespace) -> int:
    p = load_portfolio(args.portfolio)
    uc = p.get(args.use_case)
    if uc.economics is None:
        print(f"use case {uc.id} has no economics section", file=sys.stderr)
        return 2
    e = uc.economics
    bc = business_case(e)
    print(_header(p))
    print(f"Business case: {uc.name} ({uc.id}), currency {p.currency}, discount rate {e.discount_rate:.1%}\n")
    rows = [[r.year, f"{r.adoption:.0%}", f"{r.benefit:,.0f}", f"{r.build_cost + r.change_cost:,.0f}",
             f"{r.run_cost:,.0f}", f"{r.inference_cost:,.0f}", f"{r.net:,.0f}"] for r in bc.rows]
    print(table(["year", "adoption", "benefit", "build+change", "run", "inference", "net"], rows))
    payback = "not reached" if bc.payback_years is None else f"{bc.payback_years:.2f} years"
    irr = "n/a" if bc.irr is None else f"{bc.irr:.1%}"
    print(f"\nNPV: {bc.npv:,.0f}   ROI: {bc.roi:.1%}   Payback: {payback}   IRR: {irr}")
    bars = tornado(e)
    if bars:
        print("\nTornado (NPV at low / high value of one driver):")
        print(table(["driver", "npv low", "npv high", "swing"],
                    [[b.driver, f"{b.npv_low:,.0f}", f"{b.npv_high:,.0f}", f"{b.swing:,.0f}"] for b in bars]))
    mc = monte_carlo(e, n=args.samples, seed=args.seed)
    pb = "n/a" if mc.payback_p50 is None else f"{mc.payback_p50:.2f} years"
    print(f"\nMonte Carlo ({mc.n} runs, seed {mc.seed}, triangular ranges):")
    print(f"  NPV P10 / P50 / P90: {mc.npv_p10:,.0f} / {mc.npv_p50:,.0f} / {mc.npv_p90:,.0f}")
    print(f"  Mean NPV: {mc.npv_mean:,.0f}   P(NPV < 0): {mc.prob_negative_npv:.1%}")
    print(f"  Median payback: {pb}   P(no payback in horizon): {mc.prob_no_payback:.1%}")
    return 0


def _print_triage(title: str, result: TriageResult) -> None:
    print(f"{title}\n  Indicative tier: {result.tier.upper()}")
    for r in result.reasons:
        print(f"  Reason: {r}")
    print("  Obligations checklist:")
    for o in result.obligations:
        print(f"    [ ] {o.text} ({o.reference})")


def _short_reason(result: TriageResult) -> str:
    """Question id and reference of the first reason, for the overview table."""
    if result.tier == "minimal" or not result.reasons:
        return "-"
    head = result.reasons[0].split(", keyword")[0].split(", answered")[0]
    return head + ")" if head.count("(") > head.count(")") else head


def cmd_triage(args: argparse.Namespace) -> int:
    if args.describe:
        answers = {}
        for item in args.yes or []:
            if item not in ALL_QUESTION_IDS:
                print(f"unknown question id: {item}", file=sys.stderr)
                return 2
            answers[item] = True
        _print_triage("Ad-hoc use case", triage(args.describe, answers, args.role))
    else:
        if not args.portfolio:
            print("provide a portfolio file or --describe", file=sys.stderr)
            return 2
        p = load_portfolio(args.portfolio)
        print(_header(p) + "\n")
        results = triage_portfolio(p)
        if args.use_case:
            _print_triage(f"{args.use_case}: {p.get(args.use_case).name}", results[args.use_case])
        else:
            rows = [[u.id, u.name, results[u.id].tier, len(results[u.id].obligations),
                     _short_reason(results[u.id])] for u in p.use_cases]
            print(table(["id", "use case", "tier", "obligations", "first reason"], rows))
    print(f"\nDisclaimer: {DISCLAIMER}")
    return 0


def cmd_fairness(args: argparse.Namespace) -> int:
    data = load_predictions(args.predictions, args.y_true, args.y_pred, args.group, args.y_score)
    perf = performance_metrics(data)
    rep = fairness_report(data, bootstrap=args.bootstrap, seed=args.seed)
    print(f"File: {args.predictions} ({perf.n} rows)\n")
    print(f"Accuracy {perf.accuracy:.3f}  Precision {perf.precision:.3f}  Recall {perf.recall:.3f}  "
          f"F1 {perf.f1:.3f}  Specificity {perf.specificity:.3f}")
    if perf.roc_auc is not None:
        print(f"ROC AUC {perf.roc_auc:.3f}  Brier {perf.brier:.3f}")
    print(f"Confusion: TP={perf.tp} FP={perf.fp} TN={perf.tn} FN={perf.fn}\n")
    print(table(["group", "n", "selection rate", "TPR", "FPR", "accuracy"],
                [[g.group, g.n, f"{g.selection_rate:.3f}", f"{g.tpr:.3f}", f"{g.fpr:.3f}", f"{g.accuracy:.3f}"]
                 for g in rep.groups]))
    print(f"\nDemographic parity difference: {rep.demographic_parity_difference:.3f}")
    if rep.dpd_ci95:
        print(f"  bootstrap 95% interval ({args.bootstrap} resamples, seed {args.seed}): "
              f"[{rep.dpd_ci95[0]:.3f}, {rep.dpd_ci95[1]:.3f}]")
    print(f"Equalised odds difference:     {rep.equalised_odds_difference:.3f} "
          f"(TPR gap {rep.tpr_difference:.3f}, FPR gap {rep.fpr_difference:.3f})")
    print(f"Disparate impact ratio:        {rep.disparate_impact:.3f}")
    print("Flags:" if rep.flags else "Flags: none")
    for f in rep.flags:
        print(f"  - {f}")
    return 1 if (rep.flags and args.fail_on_flags) else 0


def cmd_card(args: argparse.Namespace) -> int:
    p = load_portfolio(args.portfolio)
    uc = p.get(args.use_case)
    perf = fair = None
    note = ""
    if args.predictions:
        data = load_predictions(args.predictions)
        perf, fair = performance_metrics(data), fairness_report(data, bootstrap=args.bootstrap)
        note = args.data_note or f"file {Path(args.predictions).name}"
    card = system_card(uc, triage(uc.description, uc.triage, uc.role), perf, fair, evaluation_data_note=note)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(card, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(card)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    p = load_portfolio(args.portfolio)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, content in build_reports(p, capacity=args.capacity, n_waves=args.waves).items():
        (out / name).write_text(content, encoding="utf-8")
        print(f"wrote {out / name}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="genai-gov", description="GenAI use-case governance toolkit")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("prioritise", help="score and rank the portfolio")
    s.add_argument("portfolio")
    s.set_defaults(func=cmd_prioritise)

    s = sub.add_parser("sensitivity", help="sensitivity of the ranking to the dimension weights")
    s.add_argument("portfolio")
    s.add_argument("--samples", type=int, default=2000)
    s.add_argument("--top-k", type=int, default=3)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--concentration", type=float, default=50.0)
    s.set_defaults(func=cmd_sensitivity)

    s = sub.add_parser("build-buy", help="build / buy / partner recommendation per use case")
    s.add_argument("portfolio")
    s.add_argument("--use-case", help="print the rationale for this id")
    s.set_defaults(func=cmd_build_buy)

    s = sub.add_parser("roi", help="business case with NPV, payback and Monte Carlo")
    s.add_argument("portfolio")
    s.add_argument("--use-case", required=True)
    s.add_argument("--samples", type=int, default=5000)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=cmd_roi)

    s = sub.add_parser("triage", help="indicative EU AI Act oriented risk triage")
    s.add_argument("portfolio", nargs="?")
    s.add_argument("--use-case")
    s.add_argument("--describe", help="triage a free-text description instead of a portfolio")
    s.add_argument("--yes", nargs="*", help="question ids answered yes (with --describe)")
    s.add_argument("--role", default="deployer", choices=["provider", "deployer"])
    s.set_defaults(func=cmd_triage)

    s = sub.add_parser("fairness", help="performance and group fairness metrics for a prediction CSV")
    s.add_argument("predictions")
    s.add_argument("--y-true", default="y_true")
    s.add_argument("--y-pred", default="y_pred")
    s.add_argument("--group", default="group")
    s.add_argument("--y-score", default="y_score")
    s.add_argument("--bootstrap", type=int, default=1000)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--fail-on-flags", action="store_true", help="exit with code 1 when a threshold is exceeded")
    s.set_defaults(func=cmd_fairness)

    s = sub.add_parser("card", help="generate a system card (Markdown)")
    s.add_argument("portfolio")
    s.add_argument("--use-case", required=True)
    s.add_argument("--predictions")
    s.add_argument("--data-note", default="")
    s.add_argument("--bootstrap", type=int, default=1000)
    s.add_argument("--out")
    s.set_defaults(func=cmd_card)

    s = sub.add_parser("report", help="write roadmap, risk register and RACI as Markdown")
    s.add_argument("portfolio")
    s.add_argument("--out", default="out")
    s.add_argument("--capacity", type=int, default=2)
    s.add_argument("--waves", type=int, default=3)
    s.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
