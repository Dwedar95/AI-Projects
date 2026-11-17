"""Command line entry point: ``python -m agentic_orchestrator.evals``.

Runs a suite against the offline mock provider and, when a baseline is given,
exits with status 1 if the regression gate fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..hitl import ApprovalGate
from ..mock_llm import MockLLM
from ..orchestrator import GuardrailConfig, Orchestrator
from ..toolbox import SupportBackend, build_support_registry
from .gate import GateThresholds, regression_gate
from .harness import load_suite, run_suite, write_report
from .judge import LLMJudge


def make_factory(guardrails: bool = True, gate: bool = True):
    """Factory for orchestrators over a freshly seeded synthetic backend."""

    def factory() -> Orchestrator:
        return Orchestrator(
            MockLLM(gullible=True),
            build_support_registry(SupportBackend.seeded()),
            guardrails=GuardrailConfig(scan_tool_output=guardrails),
            gate=ApprovalGate(enabled=gate),
        )

    return factory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the agent evaluation suite")
    parser.add_argument("--suite", default="evals/support_ops.jsonl")
    parser.add_argument("--baseline", help="baseline JSON to gate against")
    parser.add_argument("--write-baseline", help="write headline metrics to this path")
    parser.add_argument("--report", help="write the full per-task report to this path")
    parser.add_argument("--no-guardrails", action="store_true", help="ablation: no output scan")
    parser.add_argument("--no-gate", action="store_true", help="ablation: no approval gate")
    parser.add_argument("--max-latency-p95-ms", type=float, default=500.0)
    args = parser.parse_args(argv)

    tasks = load_suite(args.suite)
    report = run_suite(
        tasks,
        make_factory(guardrails=not args.no_guardrails, gate=not args.no_gate),
        judge=LLMJudge(MockLLM()),
    )
    print(json.dumps(report.summary(), indent=2))
    for result in report.results:
        if result.failures:
            print(f"  {result.task_id}: {'; '.join(result.failures)}")
    if args.report:
        write_report(report, args.report)
    if args.write_baseline:
        # Latency is machine dependent, so it is not part of the stored baseline.
        stable = {k: v for k, v in report.summary().items() if "latency" not in k}
        Path(args.write_baseline).write_text(json.dumps(stable, indent=2) + "\n", encoding="utf-8")
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        outcome = regression_gate(
            report, baseline, GateThresholds(max_latency_p95_ms=args.max_latency_p95_ms)
        )
        for line in outcome.checks:
            print(line)
        print("REGRESSION GATE:", "PASSED" if outcome.passed else "FAILED")
        return 0 if outcome.passed else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
