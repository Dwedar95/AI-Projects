"""End-to-end estimate: token capacity, model cost options, search sizing, total."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from archkit.capacity import ModelDemand, Workload, model_demand
from archkit.cost import BreakEven, Commitment, DeploymentOption, break_even, deployment_options
from archkit.pricing import PriceBook, load_prices
from archkit.search_sizing import (
    IndexSize,
    SearchOption,
    SearchWorkload,
    embedding_monthly_cost,
    index_size,
    search_options,
)


class PlatformWorkload(BaseModel):
    """Counts of fixed platform components, keyed by price book item."""

    model_config = ConfigDict(extra="forbid")

    components: dict[str, float] = Field(default_factory=dict)


class EstimateInput(BaseModel):
    """Schema of the workload YAML file passed to ``archkit estimate``."""

    model_config = ConfigDict(extra="forbid")

    workload: Workload
    search: SearchWorkload | None = None
    platform: PlatformWorkload = Field(default_factory=PlatformWorkload)
    ptu_commitment: Commitment = "monthly"


@dataclass(frozen=True)
class ModelEstimate:
    demand: ModelDemand
    options: list[DeploymentOption]
    break_even: BreakEven | None

    @property
    def recommended(self) -> DeploymentOption:
        return self.options[0]


@dataclass
class Estimate:
    name: str
    currency: str
    disclaimer: str
    requests_per_month: float
    peak_requests_per_minute: float
    models: dict[str, ModelEstimate]
    index: IndexSize | None = None
    search_options: list[SearchOption] = field(default_factory=list)
    embedding_cost: float = 0.0
    query_embedding_cost: float = 0.0
    platform_costs: dict[str, float] = field(default_factory=dict)

    @property
    def model_cost(self) -> float:
        return sum(m.recommended.total for m in self.models.values())

    @property
    def search_cost(self) -> float:
        return self.search_options[0].monthly_cost if self.search_options else 0.0

    @property
    def total(self) -> float:
        return (
            self.model_cost
            + self.search_cost
            + self.embedding_cost
            + self.query_embedding_cost
            + sum(self.platform_costs.values())
        )

    @property
    def cost_per_request(self) -> float:
        return self.total / self.requests_per_month

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "currency": self.currency,
            "disclaimer": self.disclaimer,
            "requests_per_month": self.requests_per_month,
            "peak_requests_per_minute": self.peak_requests_per_minute,
            "models": {
                name: {
                    "monthly_input_tokens": m.demand.monthly_input_tokens,
                    "monthly_output_tokens": m.demand.monthly_output_tokens,
                    "peak_input_tpm": m.demand.peak_input_tpm,
                    "peak_output_tpm": m.demand.peak_output_tpm,
                    "options": [asdict(o) | {"total": o.total} for o in m.options],
                    "recommended": m.recommended.strategy,
                    "break_even": asdict(m.break_even) if m.break_even else None,
                }
                for name, m in self.models.items()
            },
            "index": asdict(self.index) if self.index else None,
            "search_options": [asdict(o) | {"search_units": o.search_units} for o in self.search_options],
            "embedding_cost": self.embedding_cost,
            "query_embedding_cost": self.query_embedding_cost,
            "platform_costs": self.platform_costs,
            "total_monthly": self.total,
            "cost_per_request": self.cost_per_request,
        }

    def to_markdown(self) -> str:
        cur = self.currency
        lines = [
            f"# Capacity and cost estimate: {self.name}",
            "",
            f"> {self.disclaimer}",
            "",
            f"- Requests per month: {self.requests_per_month:,.0f}",
            f"- Peak requests per minute: {self.peak_requests_per_minute:,.1f}",
            "",
            "## Model capacity",
            "",
            "| Model class | Input tokens / month | Output tokens / month | Peak input TPM | Peak output TPM |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for name, m in self.models.items():
            d = m.demand
            lines.append(
                f"| {name} | {d.monthly_input_tokens:,.0f} | {d.monthly_output_tokens:,.0f} "
                f"| {d.peak_input_tpm:,.0f} | {d.peak_output_tpm:,.0f} |"
            )
        lines += ["", "## Deployment options per model class", ""]
        lines += [
            f"| Model class | Strategy | PTUs | PTU cost ({cur}) | PAYG cost ({cur}) | Spillover | PTU utilization "
            f"| Total ({cur}) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name, m in self.models.items():
            for o in m.options:
                mark = " (recommended)" if o is m.recommended else ""
                lines.append(
                    f"| {name} | {o.strategy}{mark} | {o.ptus} | {o.ptu_cost:,.2f} | {o.payg_cost:,.2f} "
                    f"| {o.spillover_fraction:.1%} | {o.average_utilization:.1%} | {o.total:,.2f} |"
                )
        lines += ["", "## PTU break-even", ""]
        for name, m in self.models.items():
            be = m.break_even
            if be is None:
                lines.append(f"- {name}: no PTU offer in the price book.")
            elif not be.reachable:
                lines.append(
                    f"- {name}: a PTU would need {be.utilization:.0%} average utilization to match pay-as-you-go, "
                    "so pay-as-you-go is cheaper at any volume with this token mix."
                )
            else:
                line = f"- {name}: PTU matches pay-as-you-go at {be.utilization:.1%} average utilization. "
                if be.monthly_requests is not None:
                    line += f"A peak-sized PTU deployment breaks even at {be.monthly_requests:,.0f} requests per month."
                else:
                    line += (
                        "A peak-sized PTU deployment does not break even at any scanned volume "
                        "because the daily profile keeps its average utilization below that level."
                    )
                lines.append(line)
        if self.index:
            lines += [
                "",
                "## AI Search sizing",
                "",
                f"- Chunks (after growth): {self.index.chunks:,.0f}",
                f"- Vector index: {self.index.vector_index_gb:,.2f} GB",
                f"- Total storage: {self.index.storage_gb:,.2f} GB",
                "",
            ]
            if self.search_options:
                lines += [
                    f"| Tier | Partitions | Replicas | Search units | Vector headroom | Monthly ({cur}) |",
                    "| --- | ---: | ---: | ---: | ---: | ---: |",
                ]
                for o in self.search_options:
                    lines.append(
                        f"| {o.tier} | {o.partitions} | {o.replicas} | {o.search_units} "
                        f"| {o.vector_headroom:.0%} | {o.monthly_cost:,.2f} |"
                    )
            else:
                lines.append("No tier in the price book can hold this index. Split the index or add a tier.")
        lines += ["", "## Monthly total", "", f"| Item | {cur} |", "| --- | ---: |"]
        lines.append(f"| Models (recommended options) | {self.model_cost:,.2f} |")
        lines.append(f"| AI Search | {self.search_cost:,.2f} |")
        lines.append(f"| Corpus embedding (amortised) | {self.embedding_cost:,.2f} |")
        lines.append(f"| Query embedding | {self.query_embedding_cost:,.2f} |")
        for item, cost in self.platform_costs.items():
            lines.append(f"| Platform: {item} | {cost:,.2f} |")
        lines.append(f"| **Total** | **{self.total:,.2f}** |")
        lines += ["", f"Cost per request: {self.cost_per_request:.4f} {cur}", ""]
        return "\n".join(lines)


def build_estimate(spec: EstimateInput, prices: PriceBook | None = None) -> Estimate:
    """Run the full estimate for a validated input."""
    book = prices or load_prices()
    workload = spec.workload
    models: dict[str, ModelEstimate] = {}
    for name, demand in model_demand(workload).items():
        price = book.model(name)
        options = deployment_options(demand, price, book.hours_per_month, spec.ptu_commitment)
        models[name] = ModelEstimate(
            demand, options, break_even(demand, price, book.hours_per_month, spec.ptu_commitment)
        )
    estimate = Estimate(
        name=workload.name,
        currency=book.currency,
        disclaimer=book.disclaimer,
        requests_per_month=workload.requests_per_month,
        peak_requests_per_minute=workload.peak_requests_per_minute,
        models=models,
    )
    embedding_price = book.model(workload.embedding_model).input_per_1m if workload.embedding_model else 0.0
    estimate.query_embedding_cost = (
        workload.requests_per_month * workload.embedding_tokens_per_request * embedding_price / 1e6
    )
    if spec.search is not None:
        estimate.index = index_size(spec.search)
        estimate.search_options = search_options(spec.search, book.search)
        estimate.embedding_cost = embedding_monthly_cost(spec.search, embedding_price)
    for item, count in spec.platform.components.items():
        if item not in book.platform.fixed_monthly:
            raise KeyError(f"platform component '{item}' is not in the price book")
        estimate.platform_costs[item] = count * book.platform.fixed_monthly[item]
    log_gb = workload.requests_per_month / 1e6 * book.platform.log_gb_per_million_requests
    if log_gb > 0:
        estimate.platform_costs["log_ingestion"] = log_gb * book.platform.log_ingestion_per_gb
    return estimate


def load_estimate_input(path: str | Path) -> EstimateInput:
    """Read and validate a workload YAML file."""
    return EstimateInput.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
