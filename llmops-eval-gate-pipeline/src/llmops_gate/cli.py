"""Command line interface: ``llmops register|evaluate|promote|monitor|list|benchmark``.

Exit codes: 0 success, 1 gate failure or blocked promotion, 2 critical drift
alert or usage error. That makes every command usable as a CI step.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .cost import PricingTable
from .datasets import INTENTS, read_jsonl
from .drift import load_alert_rules, monitor
from .evaluation import EvalResult, OnnxClassifierPredictor, PromptLLMPredictor, evaluate
from .gates import decision_dict, evaluate_gates, load_gate_config
from .providers import KeywordMockLLM
from .registry import Entry, PromotionBlocked, Registry, RegistryError, Stage


def build_predictor(registry: Registry, entry: Entry, llm_model: str = "small-llm"):
    """Create the predictor that matches a registry entry."""
    if entry.kind == "model":
        return OnnxClassifierPredictor(
            registry.artifact_path(entry),
            labels=entry.metadata.get("labels", INTENTS),
            embed_dim=int(entry.metadata.get("embed_dim", 256)),
        )
    return PromptLLMPredictor(entry.template or "", KeywordMockLLM(), llm_model, INTENTS)


def evaluate_entry(
    registry: Registry,
    entry: Entry,
    dataset: str | Path,
    gates: str | Path,
    baseline_metrics: dict[str, float] | None = None,
    llm_model: str = "small-llm",
) -> tuple[EvalResult, dict[str, Any]]:
    """Evaluate ``entry``, apply the gates and store the outcome on the entry.

    The baseline is, in order of preference: explicit ``baseline_metrics``, then
    the metrics of the current production version, then none (first release).
    """
    result = evaluate(
        build_predictor(registry, entry, llm_model), read_jsonl(dataset), PricingTable.example()
    )
    baseline_ref = "explicit baseline file" if baseline_metrics is not None else None
    if baseline_metrics is None:
        production = registry.in_stage(entry.kind, entry.name, Stage.PRODUCTION)
        if production is not None and production.version != entry.version:
            baseline_metrics, baseline_ref = production.metrics, production.ref
    decision = evaluate_gates(
        result.metrics, baseline_metrics, load_gate_config(gates), baseline_ref
    )
    registry.record_evaluation(entry, result.metrics, decision_dict(decision))
    return result, decision_dict(decision)


def _parse_meta(pairs: list[str]) -> dict[str, str]:
    meta = {}
    for pair in pairs:
        if "=" not in pair:
            raise RegistryError(f"--meta expects key=value, got '{pair}'")
        key, value = pair.split("=", 1)
        meta[key] = value
    return meta


def _cmd_register(registry: Registry, args: argparse.Namespace) -> int:
    meta: dict[str, Any] = {}
    if args.kind == "model":
        sidecar = Path(str(args.artifact) + ".json")
        if sidecar.exists():
            meta.update(json.loads(sidecar.read_text(encoding="utf-8")))
        meta.update(_parse_meta(args.meta))
        entry = registry.register_model(args.name, args.artifact, meta)
    else:
        template = Path(args.template_file).read_text(encoding="utf-8")
        entry = registry.register_prompt(args.name, template, _parse_meta(args.meta))
    print(f"registered {entry.ref} sha256={entry.sha256[:12]} stage={entry.stage.value}")
    return 0


def _cmd_evaluate(registry: Registry, args: argparse.Namespace) -> int:
    entry = registry.get(args.kind, args.name, args.version)
    baseline = None
    if args.baseline_metrics:
        baseline = json.loads(Path(args.baseline_metrics).read_text(encoding="utf-8"))
    result, decision = evaluate_entry(
        registry, entry, args.dataset, args.gates, baseline, args.llm_model
    )
    note = " (latency simulated by the mock provider)" if result.latency_simulated else ""
    print(f"evaluated {entry.ref} on {result.n_examples} examples{note}")
    for key, value in result.metrics.items():
        print(f"  {key:<26}{value:g}")
    print(f"baseline: {decision['baseline_ref'] or 'none (absolute thresholds only)'}")
    for check in decision["checks"]:
        print(f"  {'PASS' if check['passed'] else 'FAIL'}  {check['message']}")
    print("GATE:", "PASSED" if decision["passed"] else "FAILED")
    if args.report:
        report = {"ref": entry.ref, "result": result.model_dump(), "gate": decision}
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if decision["passed"] else 1


def _cmd_promote(registry: Registry, args: argparse.Namespace) -> int:
    entry = registry.get(args.kind, args.name, args.version)
    try:
        promoted = registry.promote(
            args.kind,
            args.name,
            entry.version,
            Stage(args.to),
            actor=args.actor,
            override_reason=args.override_reason,
        )
    except PromotionBlocked as exc:
        print(f"PROMOTION BLOCKED: {exc}")
        return 1
    print(f"promoted {promoted.ref} to {promoted.stage.value}")
    return 0


def _cmd_monitor(args: argparse.Namespace) -> int:
    report = monitor(
        read_jsonl(args.reference), read_jsonl(args.current), load_alert_rules(args.rules)
    )
    print(
        f"drift status: {report.status.upper()} "
        f"(reference n={report.n_reference}, current n={report.n_current})"
    )
    for key, value in report.metrics.items():
        print(f"  {key:<28}{value:g}")
    for alert in report.alerts:
        print(f"  ALERT [{alert.severity}] {alert.message}")
    if args.report:
        payload = {"status": report.status, **report.model_dump()}
        Path(args.report).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 2 if report.status == "critical" else 0


def _cmd_list(registry: Registry) -> int:
    for kind in ("model", "prompt"):
        for name in registry.names(kind):  # type: ignore[arg-type]
            for entry in registry.list(kind, name):  # type: ignore[arg-type]
                gate = "-" if entry.gate is None else ("pass" if entry.gate["passed"] else "fail")
                accuracy = entry.metrics.get("accuracy")
                shown = f"{accuracy:.4f}" if accuracy is not None else "-"
                print(f"{entry.ref:<34}{entry.stage.value:<12}gate={gate:<6}accuracy={shown}")
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from .optimize import format_report, run_benchmark

    report = run_benchmark(repeats=args.repeats, threads=args.threads)
    print(format_report(report))
    if args.report:
        Path(args.report).write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llmops", description=__doc__.splitlines()[0])
    parser.add_argument(
        "--registry", default=os.environ.get("LLMOPS_REGISTRY", ".registry"), help="registry root"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    register = sub.add_parser("register", help="register a model artifact or a prompt template")
    register.add_argument("kind", choices=["model", "prompt"])
    register.add_argument("--name", required=True)
    register.add_argument("--artifact", help="model file (for kind=model)")
    register.add_argument("--template-file", help="prompt template (for kind=prompt)")
    register.add_argument("--meta", action="append", default=[], metavar="KEY=VALUE")

    evaluate_cmd = sub.add_parser("evaluate", help="run the benchmark and the evaluation gates")
    evaluate_cmd.add_argument("kind", choices=["model", "prompt"])
    evaluate_cmd.add_argument("--name", required=True)
    evaluate_cmd.add_argument("--version", type=int, help="default: latest")
    evaluate_cmd.add_argument("--dataset", required=True, help="benchmark JSONL")
    evaluate_cmd.add_argument("--gates", required=True, help="gate configuration YAML")
    evaluate_cmd.add_argument("--baseline-metrics", help="JSON metrics to compare against")
    evaluate_cmd.add_argument("--llm-model", default="small-llm", help="model for prompt entries")
    evaluate_cmd.add_argument("--report", help="write a JSON report here")

    promote = sub.add_parser("promote", help="move a version to another stage")
    promote.add_argument("kind", choices=["model", "prompt"])
    promote.add_argument("--name", required=True)
    promote.add_argument("--version", type=int, help="default: latest")
    promote.add_argument("--to", required=True, choices=[s.value for s in Stage])
    promote.add_argument("--actor", default=os.environ.get("USER", "cli"))
    promote.add_argument("--override-reason", help="bypass the gate, recorded in the history")

    monitor_cmd = sub.add_parser("monitor", help="compare two inference log windows for drift")
    monitor_cmd.add_argument("--reference", required=True)
    monitor_cmd.add_argument("--current", required=True)
    monitor_cmd.add_argument("--rules", required=True, help="alert rules YAML")
    monitor_cmd.add_argument("--report")

    sub.add_parser("list", help="list registered versions")

    benchmark = sub.add_parser("benchmark", help="float32 vs int8 ONNX optimisation benchmark")
    benchmark.add_argument("--repeats", type=int, default=500)
    benchmark.add_argument("--threads", type=int, default=1)
    benchmark.add_argument("--report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "monitor":
            return _cmd_monitor(args)
        if args.command == "benchmark":
            return _cmd_benchmark(args)
        registry = Registry(args.registry)
        if args.command == "register":
            if args.kind == "model" and not args.artifact:
                raise RegistryError("register model needs --artifact")
            if args.kind == "prompt" and not args.template_file:
                raise RegistryError("register prompt needs --template-file")
            return _cmd_register(registry, args)
        if args.command == "evaluate":
            return _cmd_evaluate(registry, args)
        if args.command == "promote":
            return _cmd_promote(registry, args)
        return _cmd_list(registry)
    except (RegistryError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
