"""Command line interface: ``autonomy-kpi <command>``.

Commands: ``generate`` (synthetic logs), ``validate-odd``, ``kpis``, ``gate``
(exit code 1 on FAIL, usable in CI), ``prioritise`` and ``report``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import yaml

from .events import group_by_mission, read_log, validate_mission, write_log
from .gates import load_gate_config
from .kpis import KPI_DEFINITIONS
from .odd import load_odd, validate_odd
from .prioritisation import load_backlog, rank_backlog
from .report import analyse_build, analyse_release, fmt, render_markdown, write_reports
from .synth import BASELINE, CANDIDATE_BETTER, CANDIDATE_REGRESSED, generate_log

PROFILES = {"baseline": BASELINE, "better": CANDIDATE_BETTER, "regressed": CANDIDATE_REGRESSED}


def _cmd_generate(args: argparse.Namespace) -> int:
    events = generate_log(PROFILES[args.profile], args.missions, args.seed)
    n = write_log(events, args.out)
    print(f"wrote {n} synthetic events of {args.missions} missions to {args.out}")
    return 0


def _cmd_validate_odd(args: argparse.Namespace) -> int:
    with open(args.odd, encoding="utf-8") as fh:
        spec, problems = validate_odd(yaml.safe_load(fh))
    if spec is None:
        print(f"INVALID: {args.odd}")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"OK: ODD {spec.name!r} version {spec.version}, lighting bands {spec.lighting_bands()}")
    return 0


def _cmd_kpis(args: argparse.Namespace) -> int:
    events = read_log(args.log)
    problems = {
        mid: p for mid, evts in group_by_mission(events).items() if (p := validate_mission(evts))
    }
    if problems:
        for mid, plist in sorted(problems.items()):
            print(f"malformed mission {mid}: {'; '.join(plist)}", file=sys.stderr)
        return 2
    build = analyse_build(events, load_odd(args.odd) if args.odd else None, seed=args.seed)
    tag = " (SYNTHETIC)" if build.synthetic else ""
    print(
        f"build {build.build}{tag}: {len(build.missions)} of {build.missions_total} missions used"
    )
    for d in KPI_DEFINITIONS:
        k = build.kpis[d.name]
        print(f"  {d.name:<40} {fmt(k.value):>8} [{fmt(k.ci_low)}, {fmt(k.ci_high)}] {d.unit}")
    return 0


def _analyse(args: argparse.Namespace):
    return analyse_release(
        read_log(args.baseline),
        read_log(args.candidate),
        load_gate_config(args.gates),
        odd=load_odd(args.odd) if args.odd else None,
        backlog=load_backlog(args.backlog) if getattr(args, "backlog", None) else None,
        seed=args.seed,
    )


def _cmd_gate(args: argparse.Namespace) -> int:
    analysis = _analyse(args)
    for g in analysis.gates:
        print(
            f"{g.verdict:<4} {g.kpi:<40} base {fmt(g.baseline):>8} cand {fmt(g.candidate):>8} "
            f"p={g.p_value:.4f} ({g.test}) {g.reason}"
        )
    print(f"overall: {analysis.verdict}")
    return 1 if analysis.verdict == "FAIL" else 0


def _cmd_prioritise(args: argparse.Namespace) -> int:
    ranked = rank_backlog(load_backlog(args.backlog), args.method, args.failing, args.boost)
    for r in ranked:
        mark = f"  <- addresses {', '.join(r.addresses)}" if r.addresses else ""
        print(f"{r.rank:>2}. {r.item.id:<8} {args.method} {r.score:8.2f}  {r.item.title}{mark}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    analysis = _analyse(args)
    paths = write_reports(analysis, args.out)
    if args.print:
        print(render_markdown(analysis))
    for kind, path in paths.items():
        print(f"wrote {kind}: {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of the CLI."""
    parser = argparse.ArgumentParser(prog="autonomy-kpi", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="write a seeded synthetic mission log")
    p.add_argument("--profile", choices=sorted(PROFILES), default="baseline")
    p.add_argument("--missions", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.set_defaults(func=_cmd_generate)

    p = sub.add_parser("validate-odd", help="validate an ODD YAML file")
    p.add_argument("odd")
    p.set_defaults(func=_cmd_validate_odd)

    p = sub.add_parser("kpis", help="compute KPIs of one log")
    p.add_argument("log")
    p.add_argument("--odd")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_cmd_kpis)

    for name, func, helptext in (
        ("gate", _cmd_gate, "evaluate release gates, exit 1 on FAIL"),
        ("report", _cmd_report, "write Markdown, PNG and HTML reports"),
    ):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--baseline", required=True)
        p.add_argument("--candidate", required=True)
        p.add_argument("--gates", required=True)
        p.add_argument("--odd")
        p.add_argument("--seed", type=int, default=0)
        if name == "report":
            p.add_argument("--backlog")
            p.add_argument("--out", default="out")
            p.add_argument("--print", action="store_true", help="also print the Markdown report")
        p.set_defaults(func=func)

    p = sub.add_parser("prioritise", help="rank a backlog with RICE or WSJF")
    p.add_argument("backlog")
    p.add_argument("--method", choices=("rice", "wsjf"), default="rice")
    p.add_argument("--failing", nargs="*", default=[], help="KPIs with failing gates")
    p.add_argument("--boost", type=float, default=1.0)
    p.set_defaults(func=_cmd_prioritise)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
