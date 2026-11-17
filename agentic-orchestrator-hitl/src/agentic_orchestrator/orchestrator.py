"""The orchestrator: plan-and-execute with ReAct workers, gated tool calls and resumable runs.

Control flow for one run::

    plan  ->  for each step:  worker decides  ->  approval gate  ->  tool  ->  guardrails
                                  ^                    |
                                  |      (pause, persist, wait for human)
                                  +--------------------+

A run that needs a human decision is persisted with status ``awaiting_approval``
and returned to the caller. ``resume`` loads it, records the decision and
continues from exactly where it stopped.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

from .agents import PlannerAgent, WorkerAgent, synthesize
from .audit import AuditLog
from .guardrails import (
    StructuredOutputError,
    StructuredStats,
    sanitize_payload,
    scan_payload,
)
from .hitl import ApprovalError, ApprovalGate
from .llm import LLMProvider
from .memory import ConversationMemory, InMemoryStateStore, StateStore
from .models import (
    AgentAction,
    Observation,
    Plan,
    PlanStep,
    RiskDecision,
    RunState,
    RunStatus,
    StepRecord,
    Turn,
)
from .tools import ToolRegistry, ToolResult


@dataclass
class GuardrailConfig:
    """Switches for the content guardrails applied to tool output."""

    scan_tool_output: bool = True
    injection_threshold: float = 0.5
    max_structured_retries: int = 2


class RunNotFoundError(KeyError):
    """No run with the given id exists in the state store."""


class Orchestrator:
    """Coordinates planner, workers, tools, the approval gate and the audit log."""

    def __init__(
        self,
        llm: LLMProvider,
        registry: ToolRegistry,
        *,
        store: StateStore | None = None,
        audit: AuditLog | None = None,
        gate: ApprovalGate | None = None,
        memory: ConversationMemory | None = None,
        guardrails: GuardrailConfig | None = None,
        max_react_iterations: int = 6,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.store = store or InMemoryStateStore()
        self.audit = audit or AuditLog()
        self.gate = gate or ApprovalGate()
        self.memory = memory or ConversationMemory()
        self.guardrails = guardrails or GuardrailConfig()
        self.max_react_iterations = max_react_iterations
        retries = self.guardrails.max_structured_retries
        self.planner = PlannerAgent(llm, registry, retries)
        self.worker = WorkerAgent(llm, registry, retries)

    # ------------------------------------------------------------------ public
    def run(
        self,
        goal: str,
        mode: Literal["plan_execute", "react"] = "plan_execute",
        session_id: str | None = None,
    ) -> RunState:
        """Start a run and drive it until it completes, fails or needs approval."""
        state = RunState(goal=goal, mode=mode, session_id=session_id)
        self._log(state, "run_started", "orchestrator", {"goal": goal, "mode": mode})
        try:
            if mode == "plan_execute":
                plan, stats = self.planner.plan(goal, self.memory.context(session_id))
                self._count(state, stats)
            else:
                plan = Plan(steps=[PlanStep(id=1, description=goal)])
        except StructuredOutputError as exc:
            return self._fail(state, f"planning failed: {exc}")
        state.plan = plan
        state.records = [StepRecord(step=step) for step in plan.steps]
        self._log(state, "plan_created", "planner", {"steps": plan.model_dump()["steps"]})
        return self._advance(state)

    def resume(self, run_id: str, approved: bool, reviewer: str, reason: str = "") -> RunState:
        """Apply a human decision to a paused run and continue it."""
        state = self.get(run_id)
        request, action = state.pending_approval, state.pending_action
        if state.status is not RunStatus.AWAITING_APPROVAL or request is None or action is None:
            raise ApprovalError(f"run {run_id} is not awaiting approval")
        self.gate.decide(request, approved, reviewer, reason)
        self._log(
            state,
            "approval_decided",
            f"human:{reviewer}",
            {"approval_id": request.id, "approved": approved, "reason": reason},
        )
        if approved:
            observation = self._execute(state, action)
        else:
            observation = Observation(
                tool_name=request.tool_name,
                ok=False,
                output=f"REJECTED by human reviewer {reviewer}: {reason}",
                executed=False,
            )
        state.records[state.current_step].turns.append(Turn(action=action, observation=observation))
        state.approvals.append(request)
        state.pending_approval, state.pending_action = None, None
        state.status = RunStatus.RUNNING
        return self._advance(state)

    def get(self, run_id: str) -> RunState:
        state = self.store.get(run_id)
        if state is None:
            raise RunNotFoundError(run_id)
        return state

    def pending_approvals(self) -> list[RunState]:
        """Runs that are currently paused for a human decision."""
        return [s for s in self.store.list() if s.status is RunStatus.AWAITING_APPROVAL]

    # ---------------------------------------------------------------- internal
    def _advance(self, state: RunState) -> RunState:
        """Run the step loop until the plan is done or the run has to stop."""
        while state.current_step < len(state.records):
            record = state.records[state.current_step]
            while record.result is None:
                if len(record.turns) >= self.max_react_iterations:
                    return self._fail(
                        state,
                        f"step {record.step.id} exceeded {self.max_react_iterations} iterations",
                    )
                try:
                    action, stats = self.worker.decide(state.goal, record, self._context(state))
                except StructuredOutputError as exc:
                    return self._fail(state, f"step {record.step.id}: {exc}")
                self._count(state, stats)
                if action.action == "finish":
                    record.result = action.final_answer
                    self._log(
                        state,
                        "step_completed",
                        "worker",
                        {"step_id": record.step.id, "result": record.result},
                    )
                    break
                observation = self._gate_and_execute(state, record, action)
                if observation is None:  # paused for approval
                    return self._save(state)
                record.turns.append(Turn(action=action, observation=observation))
            state.current_step += 1
        return self._complete(state)

    def _gate_and_execute(
        self, state: RunState, record: StepRecord, action: AgentAction
    ) -> Observation | None:
        """Validate, risk-score and run one tool call. ``None`` means the run paused."""
        name = action.tool_name or ""
        spec = self.registry.get(name)
        self._log(
            state,
            "tool_requested",
            "worker",
            {"step_id": record.step.id, "tool": name, "arguments": action.arguments},
        )
        if spec is None:
            known = ", ".join(self.registry.names())
            return Observation(
                tool_name=name,
                ok=False,
                output=f"ERROR: unknown tool '{name}'. Available: {known}",
                executed=False,
            )
        outcome = self.gate.evaluate(
            state.run_id, record.step.id, spec, action.arguments, state.tainted
        )
        self._log(
            state,
            "risk_assessed",
            "gate",
            {"tool": name, **outcome.risk.model_dump(mode="json")},
        )
        if outcome.decision is RiskDecision.DENY:
            reasons = "; ".join(outcome.risk.reasons)
            return Observation(
                tool_name=name,
                ok=False,
                output=f"BLOCKED by policy (risk {outcome.risk.score:.2f}): {reasons}",
                executed=False,
            )
        if outcome.request is not None:
            state.pending_approval, state.pending_action = outcome.request, action
            state.status = RunStatus.AWAITING_APPROVAL
            self._log(
                state,
                "approval_requested",
                "gate",
                {
                    "approval_id": outcome.request.id,
                    "tool": name,
                    "arguments": action.arguments,
                    "risk": outcome.risk.score,
                },
            )
            return None
        return self._execute(state, action)

    def _execute(self, state: RunState, action: AgentAction) -> Observation:
        """Run the tool and pass its output through the injection guardrail."""
        name = action.tool_name or ""
        result: ToolResult = self.registry.execute(name, action.arguments)
        state.counters.tool_calls += 1
        flagged = False
        if result.ok and self.guardrails.scan_tool_output:
            finding = scan_payload(result.output, self.guardrails.injection_threshold)
            if finding.flagged:
                flagged = True
                state.tainted = True
                result = result.model_copy(update={"output": sanitize_payload(result.output)})
                self._log(
                    state,
                    "guardrail_triggered",
                    "guardrails",
                    {"tool": name, "score": finding.score, "rules": finding.rules},
                )
        self._log(
            state,
            "tool_executed",
            "orchestrator",
            {"tool": name, "ok": result.ok, "output": result.as_text()[:500]},
        )
        # A validation failure means the tool function never ran.
        executed = result.ok or not (result.error or "").startswith(("invalid", "unknown"))
        return Observation(
            tool_name=name,
            ok=result.ok,
            output=result.as_text(),
            executed=executed,
            flagged=flagged,
        )

    def _context(self, state: RunState) -> str:
        """Memory plus results of the steps that already finished."""
        parts = [self.memory.context(state.session_id)]
        parts.extend(
            f"Result of step {r.step.id}: {r.result}" for r in state.records if r.result is not None
        )
        return "\n".join(p for p in parts if p)

    def _complete(self, state: RunState) -> RunState:
        if len(state.records) == 1:
            state.final_answer = state.records[0].result
        else:
            response = synthesize(self.llm, state.goal, state.records)
            stats = StructuredStats()
            stats.add(response)
            self._count(state, stats)
            state.final_answer = response.text
        state.status = RunStatus.COMPLETED
        if state.session_id:
            self.memory.add(state.session_id, "user", state.goal)
            self.memory.add(state.session_id, "assistant", state.final_answer or "")
        self._log(state, "run_completed", "orchestrator", {"final_answer": state.final_answer})
        return self._save(state)

    def _fail(self, state: RunState, error: str) -> RunState:
        state.status, state.error = RunStatus.FAILED, error
        self._log(state, "run_failed", "orchestrator", {"error": error})
        return self._save(state)

    def _save(self, state: RunState) -> RunState:
        state.updated_at = time.time()
        self.store.save(state)
        return state

    def _log(self, state: RunState, event: str, actor: str, payload: dict[str, Any]) -> None:
        self.audit.append(state.run_id, event, actor, payload)

    @staticmethod
    def _count(state: RunState, stats: StructuredStats) -> None:
        counters = state.counters
        counters.llm_calls += stats.llm_calls
        counters.prompt_tokens += stats.prompt_tokens
        counters.completion_tokens += stats.completion_tokens
        counters.structured_retries += stats.retries
        counters.local_repairs += stats.local_repairs
