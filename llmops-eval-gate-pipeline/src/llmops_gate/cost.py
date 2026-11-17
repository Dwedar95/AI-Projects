"""Inference cost and latency tracking with token budget governance."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class ModelPrice(BaseModel):
    """Price per 1000 tokens in USD."""

    input_per_1k: float = Field(ge=0)
    output_per_1k: float = Field(ge=0)


class PricingTable:
    """Maps model names to prices.

    ``PricingTable.example()`` returns ILLUSTRATIVE prices chosen for the demo.
    They are not the price list of any vendor. Load real prices from your own
    contract or the provider's price sheet.
    """

    def __init__(self, prices: dict[str, ModelPrice]) -> None:
        self.prices = prices

    @classmethod
    def example(cls) -> PricingTable:
        return cls(
            {
                "large-llm": ModelPrice(input_per_1k=0.0050, output_per_1k=0.0150),
                "small-llm": ModelPrice(input_per_1k=0.0003, output_per_1k=0.0012),
                "onnx-classifier": ModelPrice(input_per_1k=0.00001, output_per_1k=0.0),
            }
        )

    def cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Cost of one request in USD. Unknown models raise ``KeyError``."""
        price = self.prices[model]
        return (input_tokens * price.input_per_1k + output_tokens * price.output_per_1k) / 1000.0

    def cheapest(self) -> str:
        return min(self.prices, key=lambda m: self.prices[m].input_per_1k)


class UsageRecord(BaseModel):
    """One metered inference request."""

    ts: float
    team: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float

    @property
    def month(self) -> str:
        return datetime.fromtimestamp(self.ts, tz=UTC).strftime("%Y-%m")


class UsageTracker:
    """Records usage in memory and optionally appends it to a JSONL file."""

    def __init__(self, pricing: PricingTable, path: str | Path | None = None) -> None:
        self.pricing = pricing
        self.path = Path(path) if path is not None else None
        self.records: list[UsageRecord] = []
        if self.path is not None and self.path.exists():
            with self.path.open(encoding="utf-8") as handle:
                self.records = [UsageRecord.model_validate_json(x) for x in handle if x.strip()]

    def record(
        self,
        team: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        ts: float | None = None,
    ) -> UsageRecord:
        entry = UsageRecord(
            ts=time.time() if ts is None else ts,
            team=team,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=self.pricing.cost(model, input_tokens, output_tokens),
        )
        self.records.append(entry)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(entry.model_dump_json() + "\n")
        return entry

    def month_totals(self, team: str, month: str) -> tuple[int, float]:
        """Tokens and USD spent by ``team`` in ``month`` (``YYYY-MM``, UTC)."""
        rows = [r for r in self.records if r.team == team and r.month == month]
        tokens = sum(r.input_tokens + r.output_tokens for r in rows)
        return tokens, float(sum(r.cost_usd for r in rows))

    def summary(self, group_by: str = "team") -> list[dict[str, Any]]:
        """Aggregate requests, tokens, cost and latency percentiles per group."""
        if group_by not in ("team", "model"):
            raise ValueError("group_by must be 'team' or 'model'")
        groups: dict[str, list[UsageRecord]] = {}
        for record in self.records:
            groups.setdefault(getattr(record, group_by), []).append(record)
        rows = []
        for key in sorted(groups):
            items = groups[key]
            latencies = np.asarray([r.latency_ms for r in items])
            cost = float(sum(r.cost_usd for r in items))
            rows.append(
                {
                    group_by: key,
                    "requests": len(items),
                    "input_tokens": sum(r.input_tokens for r in items),
                    "output_tokens": sum(r.output_tokens for r in items),
                    "cost_usd": round(cost, 6),
                    "cost_per_1k_requests_usd": round(cost / len(items) * 1000.0, 6),
                    "latency_p50_ms": round(float(np.percentile(latencies, 50)), 3),
                    "latency_p95_ms": round(float(np.percentile(latencies, 95)), 3),
                }
            )
        return rows


class BudgetAction(StrEnum):
    ALLOW = "allow"
    DOWNGRADE = "downgrade"
    DENY = "deny"


class TokenBudget(BaseModel):
    """Monthly limits for one team. ``soft_ratio`` is where downgrading starts."""

    monthly_tokens: int = Field(gt=0)
    monthly_usd: float = Field(gt=0)
    soft_ratio: float = Field(default=0.8, gt=0, le=1)


class BudgetDecision(BaseModel):
    action: BudgetAction
    model: str | None
    utilisation: float
    reason: str


class BudgetGovernor:
    """Decides before each request whether it may run and on which model.

    Below the soft limit requests run as asked. Between the soft and the hard
    limit they are downgraded to ``fallback_model`` (if it is cheaper than the
    requested one). At the hard limit they are denied. Teams without a
    configured budget are denied: an unmetered tenant is a governance gap.
    """

    def __init__(
        self,
        tracker: UsageTracker,
        budgets: dict[str, TokenBudget],
        fallback_model: str | None = None,
    ) -> None:
        self.tracker = tracker
        self.budgets = budgets
        self.fallback_model = fallback_model

    def check(
        self,
        team: str,
        model: str,
        estimated_tokens: int,
        now: float | None = None,
    ) -> BudgetDecision:
        budget = self.budgets.get(team)
        if budget is None:
            return BudgetDecision(
                action=BudgetAction.DENY, model=None, utilisation=0.0, reason="no budget configured"
            )
        month = datetime.fromtimestamp(time.time() if now is None else now, tz=UTC).strftime(
            "%Y-%m"
        )
        tokens, usd = self.tracker.month_totals(team, month)
        projected_usd = usd + self.tracker.pricing.cost(model, estimated_tokens, 0)
        utilisation = max(
            (tokens + estimated_tokens) / budget.monthly_tokens, projected_usd / budget.monthly_usd
        )
        if utilisation > 1.0:
            return BudgetDecision(
                action=BudgetAction.DENY,
                model=None,
                utilisation=round(utilisation, 4),
                reason=f"monthly budget exhausted for {month}",
            )
        fallback = self.fallback_model
        if utilisation > budget.soft_ratio and fallback and fallback != model:
            prices = self.tracker.pricing.prices
            if prices[fallback].input_per_1k < prices[model].input_per_1k:
                return BudgetDecision(
                    action=BudgetAction.DOWNGRADE,
                    model=fallback,
                    utilisation=round(utilisation, 4),
                    reason=f"above {budget.soft_ratio:.0%} of budget, routed to {fallback}",
                )
        return BudgetDecision(
            action=BudgetAction.ALLOW,
            model=model,
            utilisation=round(utilisation, 4),
            reason="within budget",
        )


def load_budgets(path: str | Path) -> dict[str, TokenBudget]:
    """Load ``{"team": {"monthly_tokens": ..., "monthly_usd": ...}}`` from JSON."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {team: TokenBudget.model_validate(cfg) for team, cfg in raw.items()}
