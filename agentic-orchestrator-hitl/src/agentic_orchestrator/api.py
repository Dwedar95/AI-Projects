"""FastAPI service exposing run, approve and audit endpoints.

``create_app`` takes a ready orchestrator so tests and deployments can inject
their own provider, tools and stores. The module-level ``app`` wires the
offline mock and the synthetic support back office for local exploration.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .hitl import ApprovalError
from .mock_llm import MockLLM
from .models import ApprovalRequest, RunState
from .orchestrator import Orchestrator, RunNotFoundError
from .toolbox import SupportBackend, build_support_registry


class RunRequest(BaseModel):
    goal: str = Field(min_length=3, max_length=2000)
    mode: Literal["plan_execute", "react"] = "plan_execute"
    session_id: str | None = Field(default=None, max_length=64)


class ApprovalDecision(BaseModel):
    approved: bool
    reviewer: str = Field(min_length=1, max_length=64)
    reason: str = Field(default="", max_length=500)


class RunView(BaseModel):
    """Client-facing projection of a run."""

    run_id: str
    status: str
    goal: str
    plan: list[str]
    executed_tools: list[str]
    final_answer: str | None
    error: str | None
    tainted: bool
    pending_approval: ApprovalRequest | None
    counters: dict[str, int]

    @classmethod
    def of(cls, state: RunState) -> RunView:
        return cls(
            run_id=state.run_id,
            status=state.status.value,
            goal=state.goal,
            plan=[s.description for s in state.plan.steps] if state.plan else [],
            executed_tools=state.executed_tools(),
            final_answer=state.final_answer,
            error=state.error,
            tainted=state.tainted,
            pending_approval=state.pending_approval,
            counters=state.counters.model_dump(),
        )


def create_app(orchestrator: Orchestrator) -> FastAPI:
    """Build the HTTP API around an orchestrator instance."""
    api = FastAPI(title="agentic-orchestrator-hitl", version="0.1.0")

    @api.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "tools": orchestrator.registry.names()}

    @api.get("/tools")
    def tools() -> list[dict[str, Any]]:
        return orchestrator.registry.schemas()

    @api.post("/runs", response_model=RunView, status_code=201)
    def start_run(request: RunRequest) -> RunView:
        return RunView.of(orchestrator.run(request.goal, request.mode, request.session_id))

    @api.get("/runs/{run_id}", response_model=RunView)
    def get_run(run_id: str) -> RunView:
        try:
            return RunView.of(orchestrator.get(run_id))
        except RunNotFoundError:
            raise HTTPException(status_code=404, detail="run not found") from None

    @api.get("/approvals", response_model=list[ApprovalRequest])
    def pending_approvals() -> list[ApprovalRequest]:
        return [s.pending_approval for s in orchestrator.pending_approvals() if s.pending_approval]

    @api.post("/runs/{run_id}/approve", response_model=RunView)
    def approve(run_id: str, decision: ApprovalDecision) -> RunView:
        try:
            state = orchestrator.resume(
                run_id, decision.approved, decision.reviewer, decision.reason
            )
        except RunNotFoundError:
            raise HTTPException(status_code=404, detail="run not found") from None
        except ApprovalError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        return RunView.of(state)

    @api.get("/audit")
    def audit(run_id: str | None = None, event: str | None = None) -> list[dict[str, Any]]:
        return [r.model_dump() for r in orchestrator.audit.records(run_id=run_id, event=event)]

    @api.get("/audit/verify")
    def audit_verify() -> dict[str, Any]:
        ok, detail = orchestrator.audit.verify()
        return {"ok": ok, "detail": detail, "records": len(orchestrator.audit)}

    return api


def default_orchestrator() -> Orchestrator:
    """Orchestrator over the offline mock and the synthetic support tools."""
    return Orchestrator(MockLLM(), build_support_registry(SupportBackend.seeded()))


app = create_app(default_orchestrator())
