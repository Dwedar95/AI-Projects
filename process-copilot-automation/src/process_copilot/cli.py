"""Command line interface: ``process-copilot <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from . import __version__
from .analytics import case_log, comparison_table, compute_metrics, recommend, roi_projection
from .engine import WorkflowEngine
from .policy import StaticApprover
from .process import load_process
from .simulation import run_simulation, simulated_reviewer
from .tools import build_default_registry


def _parse_inputs(pairs: list[str]) -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"input must be key=value, got '{pair}'")
        key, value = pair.split("=", 1)
        inputs[key] = value
    return inputs


def cmd_validate(args: argparse.Namespace) -> int:
    p = load_process(args.process)
    registry, _ = build_default_registry()
    WorkflowEngine(p, registry)
    kinds = {}
    for s in p.steps:
        kinds[s.type] = kinds.get(s.type, 0) + 1
    print(f"OK: {p.id} ({p.name})")
    print(f"  steps: {len(p.steps)} ({', '.join(f'{v} {k}' for k, v in sorted(kinds.items()))})")
    print(f"  roles: {', '.join(sorted({s.role for s in p.steps if s.role}))}")
    print(f"  systems: {', '.join(sorted({s.system for s in p.steps if s.system}))}")
    print(f"  tools allowed: {', '.join(p.allowed_tools)}")
    print(f"  approval rules: {len(p.approval_rules)}, case budget: {p.cost_budget_usd} USD, SLA: {p.sla_minutes} min")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    p = load_process(args.process)
    registry, _ = build_default_registry()
    approver = {"reviewer": simulated_reviewer(), "approve": StaticApprover(True), "deny": StaticApprover(False),
                "none": None}[args.approval]
    result = WorkflowEngine(p, registry, approver=approver).run(_parse_inputs(args.input), case_id=args.case_id)
    t = result.trace
    if args.json:
        print(json.dumps(t.to_dict(), indent=2))
        return 0
    print(f"Process: {p.name} | case {t.case_id} | status: {t.status.upper()}"
          + (f" ({t.handoff_reason})" if t.handoff_reason else ""))
    print(t.table())
    print(f"\nTotals: {t.input_tokens + t.output_tokens} tokens, model cost {t.cost_usd:.5f} USD "
          f"(budget {p.cost_budget_usd} USD), model latency {t.model_latency_ms:.0f} ms (simulated by the mock), "
          f"human time {t.human_minutes:.1f} min vs {t.manual_minutes:.1f} min manual, "
          f"interventions {t.interventions}, SLA breached: {'yes' if t.sla_breached else 'no'}")
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    p = load_process(args.process)
    traces = run_simulation(p, n=args.cases, seed=args.seed)
    m = compute_metrics(case_log(traces, seed=args.seed), hourly_rate=args.hourly_rate)
    print(f"Process: {p.name}")
    print(f"SIMULATED case log: {m.cases} cases, seed {args.seed}, mock models, fictional system data, "
          f"labour rate {m.hourly_rate:.0f} USD/h (assumption)\n")
    print(comparison_table(m))
    print(f"\nautomation rate:          {m.automation_rate:.1%}")
    print(f"intervention rate:        {m.intervention_rate:.1%} ({m.interventions_per_case:.2f} interventions per case)")
    print(f"handoff rate:             {m.handoff_rate:.1%}   approval rate: {m.approval_rate:.1%}")
    print(f"handling time saved:      {m.handling_time_saved_pct:.1%}")
    print(f"saving per case:          {m.saving_per_case:.3f} USD")
    print(f"avg tokens per case:      {m.avg_tokens:.0f}   avg model latency: {m.avg_model_latency_ms:.0f} ms (simulated)")
    print(f"SLA breach rate:          {m.sla_breach_rate:.1%}")
    if m.handoff_steps:
        print("handoffs by step:         " + ", ".join(f"{k}={v}" for k, v in m.handoff_steps.items()))
    roi = roi_projection(m, args.cases_per_month, args.platform_cost, args.one_off_cost)
    payback = "not reached" if roi.payback_months is None else f"{roi.payback_months:.1f} months"
    print(f"\nROI projection at {roi.cases_per_month} cases per month (assumed volume and costs):")
    print(f"  gross saving {roi.gross_saving_per_month:,.0f} USD/month, platform cost "
          f"{roi.platform_cost_per_month:,.0f}, net {roi.net_saving_per_month:,.0f} USD/month")
    print(f"  hours freed {roi.hours_freed_per_month:,.0f} per month, one-off cost {roi.one_off_cost:,.0f} USD, "
          f"payback {payback}")
    rec = recommend(m)
    print(f"\nPlaybook recommendation: {rec.strategy.upper()}")
    for r in rec.reasons:
        print(f"  - {r}")
    return 0


def cmd_tools(_args: argparse.Namespace) -> int:
    registry, _ = build_default_registry()
    for name in registry.names():
        tool = registry.get(name)
        print(f"{name:28s} {tool.system:10s} {'side effect' if tool.side_effect else 'read only':12s} {tool.description}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="process-copilot", description="Agent-augmented process automation reference")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("validate", help="validate a process file against the tool registry")
    s.add_argument("process")
    s.set_defaults(func=cmd_validate)

    s = sub.add_parser("run", help="run one case and print its trace")
    s.add_argument("process")
    s.add_argument("--input", action="append", default=[], metavar="KEY=VALUE")
    s.add_argument("--case-id", default="case-1")
    s.add_argument("--approval", choices=["reviewer", "approve", "deny", "none"], default="reviewer")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("simulate", help="run simulated cases and print adoption and ROI analytics")
    s.add_argument("process")
    s.add_argument("--cases", type=int, default=300)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--hourly-rate", type=float, default=45.0)
    s.add_argument("--cases-per-month", type=int, default=4000)
    s.add_argument("--platform-cost", type=float, default=1500.0)
    s.add_argument("--one-off-cost", type=float, default=60000.0)
    s.set_defaults(func=cmd_simulate)

    s = sub.add_parser("tools", help="list the registered mock tools")
    s.set_defaults(func=cmd_tools)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
