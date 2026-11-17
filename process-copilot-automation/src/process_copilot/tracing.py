"""Observability: per-step traces and per-case accounting of latency, tokens, cost and human time."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class StepTrace:
    """Everything recorded about one executed step."""

    step_id: str
    type: str
    status: str = "ok"  # ok | handoff | denied | error
    role: str = ""
    system: str = ""
    model: str = ""
    hosting: str = ""
    routing_reason: str = ""
    escalated: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model_latency_ms: float = 0.0  # reported by the provider (simulated for the mock)
    wall_ms: float = 0.0  # measured engine time for the step
    confidence: float | None = None
    tool: str = ""
    approval: str = ""  # "", "not required", "approved by ...", "denied by ..."
    human_minutes: float = 0.0
    manual_minutes: float = 0.0  # baseline handling time of this step in the manual process
    sla_minutes: float | None = None
    detail: str = ""

    @property
    def elapsed_minutes(self) -> float:
        """Processing time of the step: human handling plus model latency (queue time is not modelled)."""
        return self.human_minutes + self.model_latency_ms / 60000.0

    @property
    def sla_breached(self) -> bool:
        return self.sla_minutes is not None and self.elapsed_minutes > self.sla_minutes


@dataclass
class CaseTrace:
    """Trace of one case through a process."""

    case_id: str
    process_id: str
    status: str = "completed"  # completed | handoff
    handoff_reason: str = ""
    sla_minutes: float | None = None
    steps: list[StepTrace] = field(default_factory=list)

    @property
    def cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.steps)

    @property
    def input_tokens(self) -> int:
        return sum(s.input_tokens for s in self.steps)

    @property
    def output_tokens(self) -> int:
        return sum(s.output_tokens for s in self.steps)

    @property
    def model_latency_ms(self) -> float:
        return sum(s.model_latency_ms for s in self.steps)

    @property
    def human_minutes(self) -> float:
        return sum(s.human_minutes for s in self.steps)

    @property
    def manual_minutes(self) -> float:
        """Baseline manual handling time of the path this case took."""
        return sum(s.manual_minutes for s in self.steps)

    @property
    def interventions(self) -> int:
        """Number of points where a person had to act: approvals, human steps and the handoff."""
        return sum(1 for s in self.steps if s.human_minutes > 0 or s.approval.startswith(("approved", "denied"))
                   or s.status == "handoff")

    @property
    def automated(self) -> bool:
        """True when the case completed without any human involvement (straight-through)."""
        return self.status == "completed" and self.interventions == 0

    @property
    def elapsed_minutes(self) -> float:
        return sum(s.elapsed_minutes for s in self.steps)

    @property
    def sla_breached(self) -> bool:
        case_breach = self.sla_minutes is not None and self.elapsed_minutes > self.sla_minutes
        return case_breach or any(s.sla_breached for s in self.steps)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation including the totals."""
        return {
            "case_id": self.case_id, "process_id": self.process_id, "status": self.status,
            "handoff_reason": self.handoff_reason, "automated": self.automated, "interventions": self.interventions,
            "cost_usd": round(self.cost_usd, 6), "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
            "model_latency_ms": round(self.model_latency_ms, 1), "human_minutes": round(self.human_minutes, 2),
            "sla_breached": self.sla_breached, "steps": [asdict(s) for s in self.steps],
        }

    def table(self) -> str:
        """Plain-text table of the steps for terminal output."""
        header = ["step", "type", "status", "model", "tok in/out", "cost usd", "latency ms", "conf", "human min",
                  "approval / detail"]
        rows = [header]
        for s in self.steps:
            rows.append([
                s.step_id, s.type, s.status, s.model + ("*" if s.escalated else ""),
                f"{s.input_tokens}/{s.output_tokens}" if s.model else "-",
                f"{s.cost_usd:.5f}" if s.model else "-", f"{s.model_latency_ms:.0f}" if s.model else "-",
                "-" if s.confidence is None else f"{s.confidence:.2f}",
                f"{s.human_minutes:.1f}" if s.human_minutes else "-", "; ".join(x for x in (s.approval, s.detail) if x)[:70],
            ])
        widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
        lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip() for r in rows]
        lines.insert(1, "  ".join("-" * w for w in widths))
        return "\n".join(lines)
