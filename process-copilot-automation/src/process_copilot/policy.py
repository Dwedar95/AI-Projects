"""Policy-based human approval for action steps."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from .expressions import evaluate
from .process import ApprovalRule


@dataclass(frozen=True)
class ApprovalRequest:
    """What an approver sees: which tool call, with which arguments, and why approval is needed."""

    case_id: str
    step_id: str
    tool: str
    args: dict[str, Any]
    approver_role: str
    reason: str
    review_minutes: float


@dataclass(frozen=True)
class ApprovalDecision:
    """Outcome of an approval request."""

    approved: bool
    approver: str
    comment: str = ""


class Approver(Protocol):
    """Anything that can decide an approval request (a person behind a UI, a queue, a test double)."""

    def decide(self, request: ApprovalRequest) -> ApprovalDecision: ...


class StaticApprover:
    """Approves or denies everything. Useful for tests and dry runs."""

    def __init__(self, approve: bool = True, name: str = "static-approver"):
        self.approve, self.name = approve, name

    def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(self.approve, self.name, "auto decision")


class CallbackApprover:
    """Delegates the decision to a function, for example a simulated reviewer with a signing limit."""

    def __init__(self, fn: Callable[[ApprovalRequest], bool], name: str = "reviewer"):
        self.fn, self.name = fn, name

    def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        approved = bool(self.fn(request))
        return ApprovalDecision(approved, f"{self.name} ({request.approver_role})", "approved" if approved else "denied")


class PolicyEngine:
    """Evaluates the approval rules of a process for a concrete tool call.

    Rules are checked in order and the first match wins. A rule matches when its
    tool name equals the called tool (or is ``*``) and its ``when`` expression is
    true for the context extended with ``args``.
    """

    def __init__(self, rules: list[ApprovalRule]):
        self.rules = rules

    def requirement(self, tool: str, args: dict[str, Any], context: dict[str, Any]) -> ApprovalRule | None:
        scope = {**context, "args": args}
        for rule in self.rules:
            if rule.tool in (tool, "*") and (not rule.when or evaluate(rule.when, scope)):
                return rule
        return None
