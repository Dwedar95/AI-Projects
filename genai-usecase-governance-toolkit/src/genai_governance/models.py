"""Typed data models for a use-case portfolio and its YAML loader."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

Dimension = Literal["value", "feasibility", "risk"]


class Criterion(BaseModel):
    """One scoring criterion that belongs to a dimension.

    For the ``risk`` dimension a higher raw score means more risk. The scoring
    module inverts the risk dimension when it builds the overall score.
    """

    name: str
    dimension: Dimension
    weight: float = Field(gt=0)
    description: str = ""


class ScoringConfig(BaseModel):
    """Weights and scale used for multi-criteria scoring."""

    scale_min: float = 1.0
    scale_max: float = 5.0
    dimension_weights: dict[Dimension, float] = {"value": 0.45, "feasibility": 0.35, "risk": 0.20}
    criteria: list[Criterion]

    @model_validator(mode="after")
    def _check(self) -> ScoringConfig:
        if self.scale_max <= self.scale_min:
            raise ValueError("scale_max must be greater than scale_min")
        names = [c.name for c in self.criteria]
        if len(names) != len(set(names)):
            raise ValueError("criterion names must be unique")
        for dim in ("value", "feasibility", "risk"):
            if dim not in self.dimension_weights:
                raise ValueError(f"missing dimension weight: {dim}")
            if not any(c.dimension == dim for c in self.criteria):
                raise ValueError(f"no criteria defined for dimension: {dim}")
        if any(w < 0 for w in self.dimension_weights.values()) or sum(self.dimension_weights.values()) <= 0:
            raise ValueError("dimension weights must be non-negative and sum to a positive number")
        return self


class BuildBuyInputs(BaseModel):
    """Factors for the build / buy / partner helper, each rated 1 (low) to 5 (high)."""

    strategic_differentiation: int = Field(ge=1, le=5)
    internal_capability: int = Field(ge=1, le=5)
    market_solution_maturity: int = Field(ge=1, le=5)
    data_sensitivity: int = Field(ge=1, le=5)
    time_pressure: int = Field(ge=1, le=5)
    customisation_need: int = Field(ge=1, le=5)


class Range(BaseModel):
    """Low and high multipliers around a base value, used as a triangular distribution."""

    low: float = Field(gt=0)
    high: float = Field(gt=0)

    @model_validator(mode="after")
    def _check(self) -> Range:
        if not self.low <= 1.0 <= self.high:
            raise ValueError("range must satisfy low <= 1.0 <= high (multipliers around the base value)")
        return self


class Economics(BaseModel):
    """Inputs of the business-case model. Money is in one currency, undiscounted, per year."""

    horizon_years: int = Field(default=3, ge=1, le=10)
    discount_rate: float = Field(default=0.08, ge=0, lt=1)
    build_cost: float = Field(ge=0)
    change_management_cost: float = Field(default=0, ge=0)
    run_cost_per_year: float = Field(default=0, ge=0)
    requests_per_year: float = Field(default=0, ge=0)
    input_tokens_per_request: float = Field(default=0, ge=0)
    output_tokens_per_request: float = Field(default=0, ge=0)
    price_per_1k_input_tokens: float = Field(default=0, ge=0)
    price_per_1k_output_tokens: float = Field(default=0, ge=0)
    annual_benefit_at_full_adoption: float = Field(ge=0)
    adoption_curve: list[float] = Field(default_factory=lambda: [0.4, 0.75, 1.0])
    uncertainty: dict[str, Range] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> Economics:
        if not self.adoption_curve or any(not 0 <= a <= 1 for a in self.adoption_curve):
            raise ValueError("adoption_curve must contain values between 0 and 1")
        allowed = {"benefit", "build_cost", "run_cost", "volume", "adoption"}
        unknown = set(self.uncertainty) - allowed
        if unknown:
            raise ValueError(f"unknown uncertainty drivers: {sorted(unknown)}; allowed: {sorted(allowed)}")
        return self

    def adoption(self, year: int) -> float:
        """Adoption level in operating year ``year`` (1-based). The last value is held."""
        idx = min(year, len(self.adoption_curve)) - 1
        return self.adoption_curve[idx]


class UseCase(BaseModel):
    """A candidate AI use case with scores and optional economics and triage answers."""

    id: str
    name: str
    description: str
    business_unit: str = ""
    owner: str = ""
    scores: dict[str, float]
    depends_on: list[str] = Field(default_factory=list)
    build_buy: BuildBuyInputs | None = None
    economics: Economics | None = None
    triage: dict[str, bool] = Field(default_factory=dict)
    role: Literal["provider", "deployer"] = "deployer"


class Portfolio(BaseModel):
    """A set of use cases plus the scoring configuration used to rank them."""

    name: str
    fictional: bool = False
    currency: str = "EUR"
    scoring: ScoringConfig
    use_cases: list[UseCase]

    @model_validator(mode="after")
    def _check(self) -> Portfolio:
        ids = [u.id for u in self.use_cases]
        if len(ids) != len(set(ids)):
            raise ValueError("use case ids must be unique")
        crit = {c.name for c in self.scoring.criteria}
        lo, hi = self.scoring.scale_min, self.scoring.scale_max
        for uc in self.use_cases:
            missing = crit - set(uc.scores)
            if missing:
                raise ValueError(f"use case {uc.id} is missing scores for: {sorted(missing)}")
            for name, val in uc.scores.items():
                if name not in crit:
                    raise ValueError(f"use case {uc.id} has a score for unknown criterion: {name}")
                if not lo <= val <= hi:
                    raise ValueError(f"use case {uc.id} score {name}={val} is outside [{lo}, {hi}]")
            for dep in uc.depends_on:
                if dep not in ids:
                    raise ValueError(f"use case {uc.id} depends on unknown id: {dep}")
        return self

    def get(self, use_case_id: str) -> UseCase:
        """Return a use case by id or raise ``KeyError``."""
        for uc in self.use_cases:
            if uc.id == use_case_id:
                return uc
        raise KeyError(f"unknown use case id: {use_case_id}")


def load_portfolio(path: str | Path) -> Portfolio:
    """Load and validate a portfolio YAML file."""
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Portfolio.model_validate(data)
