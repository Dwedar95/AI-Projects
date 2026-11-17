"""Typed data contracts shared by agents, tools, the approval gate and the API."""

from __future__ import annotations

import time
import uuid
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


def new_id(prefix: str) -> str:
    """Return a short unique identifier such as ``run_3f9a1c2b4d5e``."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Message(BaseModel):
    """One chat message sent to an LLM provider."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str


class PlanStep(BaseModel):
    """A single unit of work produced by the planner."""

    id: int = Field(ge=1)
    description: str = Field(min_length=1)
    tool_hint: str | None = Field(default=None, description="Tool the planner expects to be used")


class Plan(BaseModel):
    """Structured planner output: an ordered list of steps."""

    steps: list[PlanStep] = Field(min_length=1, max_length=12)


class AgentAction(BaseModel):
    """Structured ReAct output of a worker: call a tool or finish the step."""

    thought: str = ""
    action: Literal["tool", "finish"]
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    final_answer: str | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> AgentAction:
        if self.action == "tool" and not self.tool_name:
            raise ValueError("action 'tool' requires tool_name")
        if self.action == "finish" and not self.final_answer:
            raise ValueError("action 'finish' requires final_answer")
        return self


class Observation(BaseModel):
    """What the worker sees after a tool request was processed."""

    tool_name: str
    ok: bool
    output: str
    executed: bool = Field(description="True only if the tool function really ran")
    flagged: bool = Field(default=False, description="Injection heuristics fired on the output")


class Turn(BaseModel):
    """One ReAct iteration: the action taken and the resulting observation."""

    action: AgentAction
    observation: Observation


class StepRecord(BaseModel):
    """Execution trace of one plan step."""

    step: PlanStep
    turns: list[Turn] = Field(default_factory=list)
    result: str | None = None


class RiskDecision(StrEnum):
    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


class RiskAssessment(BaseModel):
    """Outcome of scoring one proposed tool call."""

    score: float = Field(ge=0.0, le=1.0)
    decision: RiskDecision
    reasons: list[str] = Field(default_factory=list)


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ApprovalRequest(BaseModel):
    """A high-risk action waiting for, or resolved by, a human reviewer."""

    id: str = Field(default_factory=lambda: new_id("apr"))
    run_id: str
    step_id: int
    tool_name: str
    arguments: dict[str, Any]
    risk: RiskAssessment
    status: ApprovalStatus = ApprovalStatus.PENDING
    reviewer: str | None = None
    reason: str = ""
    created_at: float = Field(default_factory=time.time)
    decided_at: float | None = None


class RunStatus(StrEnum):
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"


class RunCounters(BaseModel):
    """Per-run usage counters used by the evaluation harness."""

    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls: int = 0
    structured_retries: int = 0
    local_repairs: int = 0


class RunState(BaseModel):
    """Complete, serialisable state of one orchestration run.

    Everything needed to resume after a human approval lives here, so a run can
    be paused in one process and continued in another.
    """

    run_id: str = Field(default_factory=lambda: new_id("run"))
    goal: str
    mode: Literal["plan_execute", "react"] = "plan_execute"
    session_id: str | None = None
    status: RunStatus = RunStatus.RUNNING
    plan: Plan | None = None
    records: list[StepRecord] = Field(default_factory=list)
    current_step: int = 0
    pending_action: AgentAction | None = None
    pending_approval: ApprovalRequest | None = None
    approvals: list[ApprovalRequest] = Field(default_factory=list)
    tainted: bool = Field(default=False, description="Untrusted content entered the context")
    final_answer: str | None = None
    error: str | None = None
    counters: RunCounters = Field(default_factory=RunCounters)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    def executed_tools(self) -> list[str]:
        """Names of tools whose functions really ran, in order."""
        return [
            t.observation.tool_name for r in self.records for t in r.turns if t.observation.executed
        ]
