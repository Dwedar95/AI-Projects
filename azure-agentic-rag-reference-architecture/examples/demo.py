"""Runnable demo: estimate a workload, assess two architectures, validate the Bicep.

Run from the repository root: ``python examples/demo.py``
"""

from __future__ import annotations

from pathlib import Path

from archkit.bicep import validate_directory
from archkit.checklist import assess, load_architecture
from archkit.estimate import build_estimate, load_estimate_input

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    print("=" * 78)
    print("1. Capacity and cost estimate (ILLUSTRATIVE placeholder prices, not Azure prices)")
    print("=" * 78)
    estimate = build_estimate(load_estimate_input(ROOT / "examples" / "workload.yaml"))
    print(f"workload: {estimate.name}")
    print(f"requests per month: {estimate.requests_per_month:,.0f}")
    print(f"peak requests per minute: {estimate.peak_requests_per_minute:,.1f}")
    for name, model in estimate.models.items():
        demand = model.demand
        print(f"\n[{name}] peak TPM in/out: {demand.peak_input_tpm:,.0f} / {demand.peak_output_tpm:,.0f}")
        for option in model.options:
            mark = "*" if option is model.recommended else " "
            print(
                f"  {mark} {option.strategy:<6} ptus={option.ptus:<5} total={option.total:>12,.2f} "
                f"spillover={option.spillover_fraction:>6.1%} ptu_utilization={option.average_utilization:>6.1%}"
            )
        if model.break_even:
            print(f"    PTU break-even utilization: {model.break_even.utilization:.1%}")
    if estimate.index and estimate.search_options:
        best = estimate.search_options[0]
        print(
            f"\nAI Search: {estimate.index.chunks:,.0f} chunks, vector index {estimate.index.vector_index_gb:.2f} GB, "
            f"storage {estimate.index.storage_gb:.2f} GB"
        )
        print(
            f"  cheapest feasible: tier={best.tier} partitions={best.partitions} replicas={best.replicas} "
            f"monthly={best.monthly_cost:,.2f}"
        )
    print(f"\nmonthly total: {estimate.total:,.2f} {estimate.currency}")
    print(f"cost per request: {estimate.cost_per_request:.4f} {estimate.currency}")

    print()
    print("=" * 78)
    print("2. Well-Architected style assessment")
    print("=" * 78)
    for file_name in ("architecture.yaml", "architecture-weak.yaml"):
        assessment = assess(load_architecture(ROOT / "examples" / file_name))
        pillars = ", ".join(f"{p.pillar}={p.score:.0f}" for p in assessment.pillars)
        print(f"\n{assessment.name}: gate={'PASS' if assessment.passed else 'FAIL'} overall={assessment.overall:.1f}")
        print(f"  pillars: {pillars}")
        print(f"  gaps: {len(assessment.gaps)} (critical: {len(assessment.critical_gaps)})")
        for gap in assessment.gaps[:4]:
            print(f"    - {gap.rule.id} [{gap.rule.severity}] {gap.rule.title} ({gap.status.value})")

    print()
    print("=" * 78)
    print("3. Static Bicep policy validation of infra/")
    print("=" * 78)
    report = validate_directory(ROOT / "infra")
    print(f"files={len(report.files)} resources={report.resources_checked} rules={len(report.rules)}")
    print(f"errors={len(report.errors)} warnings={len(report.warnings)} suppressed={len(report.suppressed)}")
    print(f"result: {'PASS' if report.passed else 'FAIL'}")


if __name__ == "__main__":
    main()
