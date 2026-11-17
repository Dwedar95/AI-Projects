"""Typed loader for the illustrative price and capacity configuration.

The bundled file ``data/illustrative_prices.yaml`` contains placeholders, not
Azure list prices. Pass your own file to :func:`load_prices` for real work.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PtuSpec(_Strict):
    """Provisioned throughput characteristics of one model class."""

    input_tpm_per_ptu: float = Field(gt=0)
    output_token_weight: float = Field(ge=1)
    min_ptus: int = Field(ge=1)
    increment: int = Field(ge=1)
    hourly_per_ptu: float = Field(gt=0)
    monthly_reservation_per_ptu: float = Field(gt=0)


class ModelPrice(_Strict):
    """Pay-as-you-go token prices per one million tokens."""

    input_per_1m: float = Field(ge=0)
    cached_input_per_1m: float | None = Field(default=None, ge=0)
    output_per_1m: float = Field(default=0.0, ge=0)
    ptu: PtuSpec | None = None

    @property
    def cached_price(self) -> float:
        """Cached input price, falling back to the normal input price."""
        return self.input_per_1m if self.cached_input_per_1m is None else self.cached_input_per_1m


class SearchTier(_Strict):
    storage_gb_per_partition: float = Field(gt=0)
    vector_gb_per_partition: float = Field(gt=0)
    max_partitions: int = Field(ge=1)
    max_replicas: int = Field(ge=1)
    monthly_per_search_unit: float = Field(gt=0)
    qps_per_replica: float = Field(gt=0)


class SearchPricing(_Strict):
    allowed_partition_counts: list[int]
    max_search_units: int = Field(ge=1)
    tiers: dict[str, SearchTier]

    @model_validator(mode="after")
    def _sorted_partitions(self) -> SearchPricing:
        self.allowed_partition_counts = sorted(set(self.allowed_partition_counts))
        return self


class PlatformPricing(_Strict):
    fixed_monthly: dict[str, float] = Field(default_factory=dict)
    log_ingestion_per_gb: float = Field(default=0.0, ge=0)
    log_gb_per_million_requests: float = Field(default=0.0, ge=0)


class PriceBook(_Strict):
    """All unit prices and capacity figures used by the estimator."""

    disclaimer: str
    currency: str = "USD"
    hours_per_month: float = Field(default=730, gt=0)
    models: dict[str, ModelPrice]
    search: SearchPricing
    platform: PlatformPricing = Field(default_factory=PlatformPricing)

    def model(self, name: str) -> ModelPrice:
        """Return the price entry for a model class or raise a clear error."""
        if name not in self.models:
            known = ", ".join(sorted(self.models))
            raise KeyError(f"model class '{name}' is not in the price book (known: {known})")
        return self.models[name]


def load_prices(path: str | Path | None = None) -> PriceBook:
    """Load a price book. Without a path the bundled illustrative file is used."""
    if path is None:
        text = resources.files("archkit.data").joinpath("illustrative_prices.yaml").read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    return PriceBook.model_validate(yaml.safe_load(text))
