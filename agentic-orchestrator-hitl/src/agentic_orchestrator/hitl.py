"""Human-in-the-loop approval gate.

The gate sits between "the agent wants to call a tool" and "the tool runs". It
asks the risk scorer for an assessment and either lets the call through, blocks
it outright, or creates an :class:`ApprovalRequest` that pauses the run until a
human decides.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .models import ApprovalRequest, ApprovalStatus, RiskAssessment, RiskDecision
from .risk import RiskScorer
from .tools import ToolSpec


class ApprovalError(Exception):
    """Invalid approval operation, for example deciding a request twice."""


@dataclass
class GateOutcome:
    """Decision for one proposed tool call."""

    risk: RiskAssessment
    request: ApprovalRequest | None = None

    @property
    def decision(self) -> RiskDecision:
        return self.risk.decision


class ApprovalGate:
    """Risk-scored gate in front of tool execution.

    ``enabled=False`` turns the gate into a pass-through. That exists only so the
    evaluation harness can measure what the gate contributes (an ablation); it
    is not a configuration to deploy.
    """

    def __init__(self, scorer: RiskScorer | None = None, enabled: bool = True) -> None:
        self.scorer = scorer or RiskScorer()
        self.enabled = enabled

    def evaluate(
        self,
        run_id: str,
        step_id: int,
        spec: ToolSpec,
        arguments: dict[str, Any],
        tainted: bool,
    ) -> GateOutcome:
        """Assess a call and create an approval request when one is needed."""
        risk = self.scorer.assess(spec, arguments, tainted)
        if not self.enabled:
            bypassed = risk.model_copy(
                update={
                    "decision": RiskDecision.ALLOW,
                    "reasons": risk.reasons + ["approval gate disabled"],
                }
            )
            return GateOutcome(risk=bypassed)
        if risk.decision is RiskDecision.REQUIRE_APPROVAL:
            request = ApprovalRequest(
                run_id=run_id,
                step_id=step_id,
                tool_name=spec.name,
                arguments=arguments,
                risk=risk,
            )
            return GateOutcome(risk=risk, request=request)
        return GateOutcome(risk=risk)

    @staticmethod
    def decide(
        request: ApprovalRequest, approved: bool, reviewer: str, reason: str = ""
    ) -> ApprovalRequest:
        """Record a human decision. A request can be decided exactly once."""
        if request.status is not ApprovalStatus.PENDING:
            raise ApprovalError(f"approval {request.id} is already {request.status.value}")
        if not reviewer.strip():
            raise ApprovalError("a reviewer identity is required")
        if not approved and not reason.strip():
            raise ApprovalError("a rejection needs a reason")
        request.status = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
        request.reviewer = reviewer
        request.reason = reason
        request.decided_at = time.time()
        return request
