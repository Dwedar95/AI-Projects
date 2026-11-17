"""Token throughput model for an agentic RAG workload.

A user request triggers one or more LLM calls (planning, tool selection,
answer synthesis). Requests are routed to model classes by share. Demand is
spread over the day with an hourly profile, and the peak minute inside the
busiest hour is approximated with a burst factor.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Share of daily traffic per hour for an office-hours workload (sums to 1.0).
BUSINESS_HOURS_PROFILE: tuple[float, ...] = (
    0.005, 0.005, 0.005, 0.005, 0.005, 0.010, 0.020, 0.045,
    0.080, 0.100, 0.105, 0.100, 0.075, 0.090, 0.100, 0.090,
    0.070, 0.040, 0.020, 0.010, 0.005, 0.005, 0.005, 0.005,
)  # fmt: skip


class ModelRoute(BaseModel):
    """Traffic share and token shape of the calls sent to one model class."""

    model_config = ConfigDict(extra="forbid")

    model: str
    share: float = Field(gt=0, le=1, description="Fraction of requests routed to this model class.")
    llm_calls_per_request: float = Field(default=1.0, gt=0)
    prompt_tokens_per_call: float = Field(gt=0)
    completion_tokens_per_call: float = Field(ge=0)
    cached_prompt_fraction: float = Field(default=0.0, ge=0, le=1)


class Workload(BaseModel):
    """Traffic description used by the capacity and cost estimator."""

    model_config = ConfigDict(extra="forbid")

    name: str
    requests_per_day: float = Field(gt=0)
    days_per_month: float = Field(default=30, gt=0, le=31)
    hourly_profile: list[float] | None = Field(default=None, description="24 weights, normalised on load.")
    burst_factor: float = Field(default=1.5, ge=1, description="Peak minute rate divided by the peak hour average.")
    routes: list[ModelRoute] = Field(min_length=1)
    embedding_model: str | None = None
    embedding_tokens_per_request: float = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate(self) -> Workload:
        total_share = sum(route.share for route in self.routes)
        if abs(total_share - 1.0) > 1e-6:
            raise ValueError(f"route shares must sum to 1.0, got {total_share:.4f}")
        profile = self.hourly_profile
        if profile is not None and (len(profile) != 24 or min(profile) < 0 or sum(profile) <= 0):
            raise ValueError("hourly_profile must hold 24 non-negative weights with a positive sum")
        return self

    @property
    def profile(self) -> list[float]:
        """Normalised share of daily requests per hour of the day."""
        weights = self.hourly_profile or list(BUSINESS_HOURS_PROFILE)
        total = sum(weights)
        return [w / total for w in weights]

    @property
    def requests_per_month(self) -> float:
        return self.requests_per_day * self.days_per_month

    def hourly_requests(self) -> list[float]:
        """Requests in each hour of an average day."""
        return [self.requests_per_day * share for share in self.profile]

    @property
    def peak_requests_per_minute(self) -> float:
        """Requests per minute in the busiest minute of the busiest hour."""
        return max(self.hourly_requests()) / 60.0 * self.burst_factor


@dataclass(frozen=True)
class ModelDemand:
    """Token demand of one model class derived from a workload."""

    model: str
    monthly_requests: float
    monthly_input_tokens: float
    monthly_cached_input_tokens: float
    monthly_output_tokens: float
    hourly_input_tpm: tuple[float, ...]
    hourly_output_tpm: tuple[float, ...]
    burst_factor: float
    days_per_month: float

    @property
    def monthly_uncached_input_tokens(self) -> float:
        return self.monthly_input_tokens - self.monthly_cached_input_tokens

    @property
    def peak_input_tpm(self) -> float:
        """Input tokens per minute in the peak minute."""
        return max(self.hourly_input_tpm) * self.burst_factor

    @property
    def peak_output_tpm(self) -> float:
        """Output tokens per minute in the peak minute."""
        return max(self.hourly_output_tpm) * self.burst_factor

    @property
    def peak_total_tpm(self) -> float:
        return self.peak_input_tpm + self.peak_output_tpm

    def weighted_hourly_tpm(self, output_token_weight: float) -> list[float]:
        """Average input-equivalent tokens per minute for each hour of the day."""
        return [i + output_token_weight * o for i, o in zip(self.hourly_input_tpm, self.hourly_output_tpm, strict=True)]

    def weighted_peak_tpm(self, output_token_weight: float) -> float:
        """Peak-minute demand in input-equivalent tokens per minute."""
        return max(self.weighted_hourly_tpm(output_token_weight)) * self.burst_factor

    def scaled(self, factor: float) -> ModelDemand:
        """Return the demand of the same traffic shape at ``factor`` times the volume."""
        return ModelDemand(
            model=self.model,
            monthly_requests=self.monthly_requests * factor,
            monthly_input_tokens=self.monthly_input_tokens * factor,
            monthly_cached_input_tokens=self.monthly_cached_input_tokens * factor,
            monthly_output_tokens=self.monthly_output_tokens * factor,
            hourly_input_tpm=tuple(v * factor for v in self.hourly_input_tpm),
            hourly_output_tpm=tuple(v * factor for v in self.hourly_output_tpm),
            burst_factor=self.burst_factor,
            days_per_month=self.days_per_month,
        )


def model_demand(workload: Workload) -> dict[str, ModelDemand]:
    """Compute the token demand per model class. Routes to the same class are summed."""
    hourly_requests = workload.hourly_requests()
    totals: dict[str, dict[str, float]] = {}
    for route in workload.routes:
        entry = totals.setdefault(route.model, {"req": 0.0, "in": 0.0, "cached": 0.0, "out": 0.0})
        calls = route.share * route.llm_calls_per_request
        entry["req"] += route.share
        entry["in"] += calls * route.prompt_tokens_per_call
        entry["cached"] += calls * route.prompt_tokens_per_call * route.cached_prompt_fraction
        entry["out"] += calls * route.completion_tokens_per_call
    demand: dict[str, ModelDemand] = {}
    for model, per_request in totals.items():
        demand[model] = ModelDemand(
            model=model,
            monthly_requests=workload.requests_per_month * per_request["req"],
            monthly_input_tokens=workload.requests_per_month * per_request["in"],
            monthly_cached_input_tokens=workload.requests_per_month * per_request["cached"],
            monthly_output_tokens=workload.requests_per_month * per_request["out"],
            hourly_input_tpm=tuple(r * per_request["in"] / 60.0 for r in hourly_requests),
            hourly_output_tpm=tuple(r * per_request["out"] / 60.0 for r in hourly_requests),
            burst_factor=workload.burst_factor,
            days_per_month=workload.days_per_month,
        )
    return demand
