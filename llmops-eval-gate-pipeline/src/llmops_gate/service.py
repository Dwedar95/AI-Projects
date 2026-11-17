"""FastAPI inference service for the production model in the registry.

Each request passes the token budget governor, is served by the ONNX classifier
and is metered (tokens, cost, latency). Configuration via environment:

- ``LLMOPS_REGISTRY``: registry root (default ``.registry``)
- ``LLMOPS_MODEL``: model name (default ``intent-classifier``)
- ``LLMOPS_BUDGETS``: optional JSON file with per-team budgets
- ``LLMOPS_USAGE_LOG``: optional JSONL path for usage records
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .cost import (
    BudgetAction,
    BudgetGovernor,
    PricingTable,
    TokenBudget,
    UsageTracker,
    load_budgets,
)
from .datasets import INTENTS
from .embedding import estimate_tokens
from .evaluation import OnnxClassifierPredictor
from .registry import Registry, Stage

MODEL_PRICE_KEY = "onnx-classifier"


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    team: str = Field(default="default", min_length=1, max_length=64)


class PredictResponse(BaseModel):
    label: str
    score: float
    model_ref: str
    latency_ms: float
    budget_utilisation: float


def create_app(
    registry: Registry,
    model_name: str = "intent-classifier",
    budgets: dict[str, TokenBudget] | None = None,
    usage_log: str | None = None,
) -> FastAPI:
    """Build the service. The production model is loaded lazily and reloaded on change."""
    app = FastAPI(title="llmops-inference", version="0.1.0")
    tracker = UsageTracker(PricingTable.example(), usage_log)
    governor = BudgetGovernor(
        tracker, budgets or {"default": TokenBudget(monthly_tokens=5_000_000, monthly_usd=50.0)}
    )
    loaded: dict[str, Any] = {"ref": None, "predictor": None}

    def production_predictor() -> tuple[str, OnnxClassifierPredictor] | None:
        entry = registry.in_stage("model", model_name, Stage.PRODUCTION)
        if entry is None:
            return None
        if loaded["ref"] != entry.ref:
            if not registry.verify_artifact(entry):
                raise HTTPException(status_code=503, detail=f"{entry.ref}: artifact hash mismatch")
            loaded["predictor"] = OnnxClassifierPredictor(
                registry.artifact_path(entry),
                labels=entry.metadata.get("labels", INTENTS),
                embed_dim=int(entry.metadata.get("embed_dim", 256)),
            )
            loaded["ref"] = entry.ref
        return entry.ref, loaded["predictor"]

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> dict[str, str]:
        current = production_predictor()
        if current is None:
            raise HTTPException(status_code=503, detail="no production model")
        return {"status": "ready", "model_ref": current[0]}

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        current = production_predictor()
        if current is None:
            raise HTTPException(status_code=503, detail="no production model")
        ref, predictor = current
        tokens = estimate_tokens(request.text)
        decision = governor.check(request.team, MODEL_PRICE_KEY, tokens)
        if decision.action is BudgetAction.DENY:
            raise HTTPException(status_code=429, detail=f"budget: {decision.reason}")
        prediction = predictor.predict([request.text])[0]
        tracker.record(request.team, MODEL_PRICE_KEY, tokens, 0, prediction.latency_ms)
        return PredictResponse(
            label=prediction.label,
            score=round(prediction.score, 4),
            model_ref=ref,
            latency_ms=round(prediction.latency_ms, 4),
            budget_utilisation=decision.utilisation,
        )

    @app.get("/metrics")
    def metrics() -> dict[str, Any]:
        return {"by_team": tracker.summary("team"), "by_model": tracker.summary("model")}

    return app


def _app_from_env() -> FastAPI:
    budgets_path = os.environ.get("LLMOPS_BUDGETS")
    return create_app(
        Registry(os.environ.get("LLMOPS_REGISTRY", ".registry")),
        os.environ.get("LLMOPS_MODEL", "intent-classifier"),
        load_budgets(budgets_path) if budgets_path else None,
        os.environ.get("LLMOPS_USAGE_LOG"),
    )


app = _app_from_env()
