"""Risk scoring for proposed tool calls.

The score is additive and explainable: a base value from the tool's declared
side effect, plus modifiers for irreversibility, argument values and whether
untrusted content has entered the run context. Every contribution is recorded
as a human-readable reason that ends up in the audit log and in the approval
request shown to the reviewer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .models import RiskAssessment, RiskDecision
from .tools import SideEffect, ToolSpec

ArgumentRule = Callable[[ToolSpec, dict[str, Any]], tuple[float, str] | None]

BASE_RISK: dict[SideEffect, float] = {
    SideEffect.NONE: 0.0,
    SideEffect.WRITE: 0.25,
    SideEffect.EXTERNAL: 0.30,
    SideEffect.DESTRUCTIVE: 0.70,
}


def amount_rule(medium: float = 100.0, high: float = 1000.0) -> ArgumentRule:
    """Raise risk with the monetary ``amount`` argument."""

    def rule(spec: ToolSpec, args: dict[str, Any]) -> tuple[float, str] | None:
        amount = args.get("amount")
        if not isinstance(amount, int | float):
            return None
        if amount > high:
            return 0.40, f"amount {amount:.2f} is above {high:.0f}"
        if amount > medium:
            return 0.20, f"amount {amount:.2f} is above {medium:.0f}"
        return None

    return rule


def external_recipient_rule(internal_domains: tuple[str, ...]) -> ArgumentRule:
    """Raise risk when a message leaves the organisation's own domains."""

    def rule(spec: ToolSpec, args: dict[str, Any]) -> tuple[float, str] | None:
        recipient = args.get("to")
        if not isinstance(recipient, str) or "@" not in recipient:
            return None
        domain = recipient.rsplit("@", 1)[1].lower()
        if domain in internal_domains:
            return None
        return 0.25, f"recipient domain '{domain}' is external"

    return rule


@dataclass
class RiskPolicy:
    """Thresholds and modifiers used by :class:`RiskScorer`."""

    approval_threshold: float = 0.5
    deny_threshold: float = 0.95
    irreversible_penalty: float = 0.10
    tainted_penalty: float = 0.30
    argument_rules: list[ArgumentRule] = field(
        default_factory=lambda: [amount_rule(), external_recipient_rule(("company.example",))]
    )

    def __post_init__(self) -> None:
        if not 0.0 < self.approval_threshold <= self.deny_threshold <= 1.0:
            raise ValueError("require 0 < approval_threshold <= deny_threshold <= 1")


class RiskScorer:
    """Maps a proposed tool call to a score in [0, 1] and a decision."""

    def __init__(self, policy: RiskPolicy | None = None) -> None:
        self.policy = policy or RiskPolicy()

    def assess(self, spec: ToolSpec, arguments: dict[str, Any], tainted: bool) -> RiskAssessment:
        policy = self.policy
        score = BASE_RISK[spec.side_effect]
        reasons = [f"side effect '{spec.side_effect.value}' (base {score:.2f})"]
        has_effect = spec.side_effect is not SideEffect.NONE
        if has_effect and not spec.reversible:
            score += policy.irreversible_penalty
            reasons.append(f"irreversible (+{policy.irreversible_penalty:.2f})")
        for rule in policy.argument_rules:
            hit = rule(spec, arguments)
            if hit is not None:
                score += hit[0]
                reasons.append(f"{hit[1]} (+{hit[0]:.2f})")
        if tainted and has_effect:
            score += policy.tainted_penalty
            reasons.append(f"untrusted content in run context (+{policy.tainted_penalty:.2f})")
        score = round(min(score, 1.0), 4)
        if score >= policy.deny_threshold:
            decision = RiskDecision.DENY
        elif score >= policy.approval_threshold:
            decision = RiskDecision.REQUIRE_APPROVAL
        else:
            decision = RiskDecision.ALLOW
        return RiskAssessment(score=score, decision=decision, reasons=reasons)
