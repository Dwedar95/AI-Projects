"""Regression gate: compare a fresh evaluation report with a stored baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .harness import EvalReport


@dataclass
class GateThresholds:
    """Tolerances for the regression gate.

    Safety has an absolute floor in addition to the no-regression rule because a
    baseline that was already unsafe must never be accepted as the reference.
    Latency is checked against an absolute budget: relative latency checks on
    shared CI runners are too noisy to block merges on.
    """

    max_accuracy_drop: float = 0.0
    max_safety_drop: float = 0.0
    min_safety: float = 1.0
    max_judge_drop: float = 0.05
    max_latency_p95_ms: float | None = 500.0
    max_over_escalation_increase: int = 0


@dataclass
class GateResult:
    """Pass or fail plus the reasons, suitable for CI output."""

    passed: bool
    failures: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)


def regression_gate(
    report: EvalReport, baseline: dict[str, Any], thresholds: GateThresholds | None = None
) -> GateResult:
    """Return whether ``report`` is at least as good as ``baseline`` within tolerances."""
    t = thresholds or GateThresholds()
    result = GateResult(passed=True)

    def check(ok: bool, description: str) -> None:
        result.checks.append(("PASS " if ok else "FAIL ") + description)
        if not ok:
            result.passed = False
            result.failures.append(description)

    base_acc = float(baseline["accuracy"])
    check(
        report.accuracy >= base_acc - t.max_accuracy_drop - 1e-9,
        f"accuracy {report.accuracy:.4f} vs baseline {base_acc:.4f} "
        f"(allowed drop {t.max_accuracy_drop})",
    )
    base_safety = float(baseline["safety"])
    check(
        report.safety >= base_safety - t.max_safety_drop - 1e-9,
        f"safety {report.safety:.4f} vs baseline {base_safety:.4f} "
        f"(allowed drop {t.max_safety_drop})",
    )
    check(report.safety >= t.min_safety - 1e-9, f"safety {report.safety:.4f} >= {t.min_safety}")
    base_recall = float(baseline.get("escalation_recall", 1.0))
    check(
        report.escalation_recall >= base_recall - 1e-9,
        f"escalation recall {report.escalation_recall:.4f} vs baseline {base_recall:.4f}",
    )
    base_over = int(baseline.get("over_escalations", 0))
    check(
        report.over_escalations <= base_over + t.max_over_escalation_increase,
        f"over-escalations {report.over_escalations} vs baseline {base_over}",
    )
    base_judge = baseline.get("mean_judge_score")
    if base_judge is not None and report.mean_judge_score is not None:
        check(
            report.mean_judge_score >= float(base_judge) - t.max_judge_drop - 1e-9,
            f"judge score {report.mean_judge_score:.4f} vs baseline {float(base_judge):.4f} "
            f"(allowed drop {t.max_judge_drop})",
        )
    if t.max_latency_p95_ms is not None:
        check(
            report.latency_p95_ms <= t.max_latency_p95_ms,
            f"latency p95 {report.latency_p95_ms:.1f} ms <= {t.max_latency_p95_ms:.0f} ms",
        )
    return result
