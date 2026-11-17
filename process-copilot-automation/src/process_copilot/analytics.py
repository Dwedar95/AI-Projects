"""Adoption and ROI analytics computed from case traces.

The before / after comparison is paired per case. "Before" is the manual
handling time of the path the case took (the ``manual_minutes`` of the visited
steps). "After" is the human time that remains in the augmented workflow
(approvals, manual steps, handoffs) plus the model cost. A per-case lognormal
noise factor, drawn from a seeded generator, is applied to both sides.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field

from .tracing import CaseTrace


@dataclass(frozen=True)
class CaseRecord:
    """One row of the case log used for analytics."""

    case_id: str
    status: str
    automated: bool
    interventions: int
    approvals: int
    baseline_minutes: float
    human_minutes: float
    model_cost_usd: float
    tokens: int
    model_latency_ms: float
    sla_breached: bool
    handoff_reason: str = ""


def case_log(traces: list[CaseTrace], seed: int = 7, noise_sigma: float = 0.25) -> list[CaseRecord]:
    """Turn traces into a case log with seeded handling-time noise (same factor before and after)."""
    rng = random.Random(seed)
    records = []
    for t in traces:
        factor = math.exp(rng.gauss(0.0, noise_sigma)) if noise_sigma > 0 else 1.0
        records.append(CaseRecord(
            case_id=t.case_id, status=t.status, automated=t.automated, interventions=t.interventions,
            approvals=sum(1 for s in t.steps if s.approval.startswith(("approved", "denied"))),
            baseline_minutes=t.manual_minutes * factor, human_minutes=t.human_minutes * factor,
            model_cost_usd=t.cost_usd, tokens=t.input_tokens + t.output_tokens,
            model_latency_ms=t.model_latency_ms, sla_breached=t.sla_breached,
            handoff_reason=t.handoff_reason.split(": ", 1)[0] if t.handoff_reason else "",
        ))
    return records


@dataclass
class AdoptionMetrics:
    """Aggregated metrics of a case log."""

    cases: int
    automation_rate: float
    intervention_rate: float
    interventions_per_case: float
    handoff_rate: float
    approval_rate: float
    avg_minutes_before: float
    avg_minutes_after: float
    handling_time_saved_pct: float
    cost_per_case_before: float
    cost_per_case_after: float
    model_cost_per_case: float
    avg_tokens: float
    avg_model_latency_ms: float
    sla_breach_rate: float
    hourly_rate: float
    handoff_steps: dict[str, int] = field(default_factory=dict)

    @property
    def saving_per_case(self) -> float:
        return self.cost_per_case_before - self.cost_per_case_after


def compute_metrics(records: list[CaseRecord], hourly_rate: float = 45.0) -> AdoptionMetrics:
    """Aggregate a case log. ``hourly_rate`` is the assumed fully loaded labour cost per hour in USD."""
    n = len(records)
    if n == 0:
        raise ValueError("the case log is empty")
    before = sum(r.baseline_minutes for r in records) / n
    after = sum(r.human_minutes for r in records) / n
    model_cost = sum(r.model_cost_usd for r in records) / n
    return AdoptionMetrics(
        cases=n,
        automation_rate=sum(r.automated for r in records) / n,
        intervention_rate=sum(r.interventions > 0 for r in records) / n,
        interventions_per_case=sum(r.interventions for r in records) / n,
        handoff_rate=sum(r.status == "handoff" for r in records) / n,
        approval_rate=sum(r.approvals > 0 for r in records) / n,
        avg_minutes_before=before,
        avg_minutes_after=after,
        handling_time_saved_pct=(before - after) / before if before > 0 else 0.0,
        cost_per_case_before=before / 60.0 * hourly_rate,
        cost_per_case_after=after / 60.0 * hourly_rate + model_cost,
        model_cost_per_case=model_cost,
        avg_tokens=sum(r.tokens for r in records) / n,
        avg_model_latency_ms=sum(r.model_latency_ms for r in records) / n,
        sla_breach_rate=sum(r.sla_breached for r in records) / n,
        hourly_rate=hourly_rate,
        handoff_steps=dict(Counter(r.handoff_reason for r in records if r.handoff_reason).most_common()),
    )


@dataclass(frozen=True)
class RoiProjection:
    """Monthly projection from per-case savings and assumed volume and platform cost."""

    cases_per_month: int
    gross_saving_per_month: float
    platform_cost_per_month: float
    net_saving_per_month: float
    one_off_cost: float
    payback_months: float | None
    hours_freed_per_month: float


def roi_projection(m: AdoptionMetrics, cases_per_month: int, platform_cost_per_month: float,
                   one_off_cost: float) -> RoiProjection:
    """Project monthly savings and payback. Payback is None when the net saving is not positive."""
    gross = m.saving_per_case * cases_per_month
    net = gross - platform_cost_per_month
    return RoiProjection(
        cases_per_month=cases_per_month,
        gross_saving_per_month=gross,
        platform_cost_per_month=platform_cost_per_month,
        net_saving_per_month=net,
        one_off_cost=one_off_cost,
        payback_months=one_off_cost / net if net > 0 else None,
        hours_freed_per_month=(m.avg_minutes_before - m.avg_minutes_after) * cases_per_month / 60.0,
    )


@dataclass(frozen=True)
class Recommendation:
    """Augment-versus-rewrite recommendation with the metric-based reasons (see docs/playbook.md)."""

    strategy: str
    reasons: list[str]


# Thresholds used by the playbook. They are starting values and should be tuned per organisation.
REWRITE_MIN_AUTOMATION = 0.60
REWRITE_MAX_HANDOFF = 0.10
STABILISE_MIN_HANDOFF = 0.25
MIN_TIME_SAVED = 0.20
HIGH_APPROVAL_RATE = 0.30


def recommend(m: AdoptionMetrics) -> Recommendation:
    """Map metrics to one of: rewrite, augment, stabilise, stop."""
    reasons: list[str] = []
    if m.handoff_rate > STABILISE_MIN_HANDOFF:
        strategy = "stabilise"
        top = next(iter(m.handoff_steps), "n/a")
        reasons.append(f"handoff rate {m.handoff_rate:.0%} is above {STABILISE_MIN_HANDOFF:.0%}; "
                       f"fix the main failure point first (step '{top}')")
    elif m.handling_time_saved_pct < MIN_TIME_SAVED:
        strategy = "stop"
        reasons.append(f"handling time saved {m.handling_time_saved_pct:.0%} is below {MIN_TIME_SAVED:.0%}; "
                       "the workflow does not remove enough work to justify more investment")
    elif m.automation_rate >= REWRITE_MIN_AUTOMATION and m.handoff_rate <= REWRITE_MAX_HANDOFF:
        strategy = "rewrite"
        reasons.append(f"automation rate {m.automation_rate:.0%} >= {REWRITE_MIN_AUTOMATION:.0%} and handoff rate "
                       f"{m.handoff_rate:.0%} <= {REWRITE_MAX_HANDOFF:.0%}; redesign the process as AI-first "
                       "with people handling exceptions only")
    else:
        strategy = "augment"
        reasons.append(f"automation rate {m.automation_rate:.0%} with intervention rate {m.intervention_rate:.0%}; "
                       "keep the process shape and the human checkpoints, automate the preparation steps")
    if m.approval_rate > HIGH_APPROVAL_RATE:
        reasons.append(f"{m.approval_rate:.0%} of cases wait for an approval; review the approval thresholds")
    if m.sla_breach_rate > 0.05:
        reasons.append(f"SLA breach rate {m.sla_breach_rate:.0%}; check the manual steps that remain")
    return Recommendation(strategy, reasons)


def comparison_table(m: AdoptionMetrics) -> str:
    """Before / after table as plain text."""
    rows = [
        ("metric", "before (manual)", "after (augmented)"),
        ("handling time per case (min)", f"{m.avg_minutes_before:.2f}", f"{m.avg_minutes_after:.2f}"),
        ("labour + model cost per case (USD)", f"{m.cost_per_case_before:.3f}", f"{m.cost_per_case_after:.3f}"),
        ("model cost per case (USD)", "0.00000", f"{m.model_cost_per_case:.5f}"),
        ("cases without human involvement", "0.0%", f"{m.automation_rate:.1%}"),
        ("cases with human intervention", "100.0%", f"{m.intervention_rate:.1%}"),
    ]
    widths = [max(len(r[i]) for r in rows) for i in range(3)]
    lines = ["  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(r, widths, strict=True)))
             for r in rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)
