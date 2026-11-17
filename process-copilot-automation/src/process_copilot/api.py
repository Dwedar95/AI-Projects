"""Optional FastAPI application that exposes the example processes over HTTP.

Run with ``uvicorn process_copilot.api:create_app --factory`` (needs the ``serve`` extra).
The app uses the mock provider and the mock systems, it is a demo surface and not a
production deployment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .engine import WorkflowEngine
from .policy import StaticApprover
from .process import ProcessDefinition, load_process
from .simulation import simulated_reviewer
from .tools import build_default_registry


class RunRequest(BaseModel):
    """Body of ``POST /processes/{process_id}/run``."""

    inputs: dict[str, Any]
    case_id: str = "api-case"
    approval: str = "reviewer"  # reviewer | approve | deny | none


def create_app(process_dir: str | Path | None = None) -> FastAPI:
    """Build the app for all process files of a directory (default: the bundled examples)."""
    directory = Path(process_dir) if process_dir else Path(__file__).resolve().parents[2] / "examples" / "processes"
    processes: dict[str, ProcessDefinition] = {}
    for path in sorted(directory.glob("*.yaml")):
        process = load_process(path)
        processes[process.id] = process
    registry, _systems = build_default_registry()
    approvers = {"reviewer": simulated_reviewer(), "approve": StaticApprover(True), "deny": StaticApprover(False),
                 "none": None}
    app = FastAPI(title="process-copilot-automation", version="0.1.0")

    @app.get("/processes")
    def list_processes() -> list[dict[str, Any]]:
        return [{"id": p.id, "name": p.name, "steps": len(p.steps), "inputs": p.inputs} for p in processes.values()]

    @app.get("/tools")
    def list_tools() -> list[dict[str, Any]]:
        return [{"name": n, "system": registry.get(n).system, "side_effect": registry.get(n).side_effect}
                for n in registry.names()]

    @app.post("/processes/{process_id}/run")
    def run_process(process_id: str, body: RunRequest) -> dict[str, Any]:
        if process_id not in processes:
            raise HTTPException(status_code=404, detail=f"unknown process: {process_id}")
        if body.approval not in approvers:
            raise HTTPException(status_code=422, detail=f"approval must be one of {sorted(approvers)}")
        engine = WorkflowEngine(processes[process_id], registry, approver=approvers[body.approval])
        try:
            result = engine.run(body.inputs, case_id=body.case_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return result.trace.to_dict()

    return app
