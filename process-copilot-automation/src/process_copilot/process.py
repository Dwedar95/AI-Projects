"""Process definition models and YAML loader.

A process is an ordered list of steps. Each step names the business role that
owns it, the system it touches, its SLA and the manual handling time of the
step in the current (pre-automation) process, which the analytics module uses
as the baseline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

StepType = Literal["prompt", "rag", "action", "decision", "human"]
Task = Literal["classify", "extract", "summarise", "generate"]
END = "end"


class Branch(BaseModel):
    """One branch of a decision step: go to ``goto`` when ``when`` is true."""

    when: str
    goto: str


class ApprovalRule(BaseModel):
    """Policy rule: calls to ``tool`` need approval by ``approver_role`` when ``when`` is true.

    ``when`` is evaluated against the workflow context plus ``args`` (the rendered tool arguments).
    An empty ``when`` means the rule always applies.
    """

    tool: str
    when: str = ""
    approver_role: str
    reason: str = ""
    review_minutes: float = Field(default=2.0, ge=0)


class Step(BaseModel):
    """One workflow step. Which fields are required depends on ``type``."""

    id: str
    type: StepType
    name: str = ""
    role: str = ""
    system: str = ""
    sla_minutes: float | None = Field(default=None, gt=0)
    manual_minutes: float = Field(default=0.0, ge=0)
    output: str = ""
    next: str | None = None
    # prompt and rag steps
    task: Task | None = None
    prompt: str = ""
    input: str = ""
    labels: dict[str, list[str]] = Field(default_factory=dict)
    fields: dict[str, str] = Field(default_factory=dict)
    complexity: Literal["low", "medium", "high"] | None = None
    data_classification: Literal["public", "internal", "confidential"] = "internal"
    min_confidence: float | None = Field(default=None, ge=0, le=1)
    max_output_tokens: int = Field(default=120, gt=0)
    # rag steps
    query: str = ""
    top_k: int = Field(default=3, ge=1)
    # action steps
    tool: str = ""
    args: dict[str, Any] = Field(default_factory=dict)
    # decision steps
    branches: list[Branch] = Field(default_factory=list)
    default: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Step:
        if self.id == END:
            raise ValueError("'end' is reserved and cannot be a step id")
        if self.type == "prompt":
            if not self.task or not self.prompt:
                raise ValueError(f"prompt step {self.id} needs 'task' and 'prompt'")
            if self.task == "classify" and len(self.labels) < 2:
                raise ValueError(f"classify step {self.id} needs at least two labels")
            if self.task == "extract" and not self.fields:
                raise ValueError(f"extract step {self.id} needs 'fields'")
        if self.type == "rag" and not self.query:
            raise ValueError(f"rag step {self.id} needs 'query'")
        if self.type == "action" and not self.tool:
            raise ValueError(f"action step {self.id} needs 'tool'")
        if self.type == "decision" and (not self.branches or self.default is None):
            raise ValueError(f"decision step {self.id} needs 'branches' and 'default'")
        if not self.output:
            self.output = self.id
        return self


class ProcessDefinition(BaseModel):
    """A business process with its steps, SLA, cost budget, tool allowlist and approval policy.

    ``handoff_manual_minutes`` is the time a person needs to finish a case that the
    workflow hands over (low confidence, denied approval, tool error or budget).
    """

    id: str
    name: str
    description: str = ""
    owner: str = ""
    sla_minutes: float | None = Field(default=None, gt=0)
    cost_budget_usd: float = Field(default=0.05, gt=0)
    knowledge_base: str | None = None
    handoff_manual_minutes: float = Field(default=10.0, ge=0)
    inputs: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    approval_rules: list[ApprovalRule] = Field(default_factory=list)
    steps: list[Step] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> ProcessDefinition:
        ids = [s.id for s in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("step ids must be unique")
        valid = set(ids) | {END}
        for s in self.steps:
            targets = [b.goto for b in s.branches] + [t for t in (s.default, s.next) if t]
            for t in targets:
                if t not in valid:
                    raise ValueError(f"step {s.id} points to unknown step '{t}'")
            if s.type == "action" and s.tool not in self.allowed_tools:
                raise ValueError(f"step {s.id} calls tool '{s.tool}' which is not in 'allowed_tools'")
            if s.type == "rag" and not self.knowledge_base:
                raise ValueError(f"rag step {s.id} needs a process-level 'knowledge_base'")
        return self

    def step(self, step_id: str) -> Step:
        """Return a step by id."""
        for s in self.steps:
            if s.id == step_id:
                return s
        raise KeyError(step_id)

    def successor(self, step_id: str) -> str:
        """Id of the step that follows ``step_id`` by default (its ``next`` or the following step)."""
        s = self.step(step_id)
        if s.next:
            return s.next
        idx = self.steps.index(s)
        return self.steps[idx + 1].id if idx + 1 < len(self.steps) else END

    @property
    def manual_minutes(self) -> float:
        """Total manual handling time of all steps in the baseline process."""
        return sum(s.manual_minutes for s in self.steps)


def load_process(path: str | Path) -> ProcessDefinition:
    """Load and validate a process YAML file. A relative knowledge base path is resolved to the file."""
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    process = ProcessDefinition.model_validate(data)
    if process.knowledge_base and not Path(process.knowledge_base).is_absolute():
        process.knowledge_base = str((path.parent / process.knowledge_base).resolve())
    return process
