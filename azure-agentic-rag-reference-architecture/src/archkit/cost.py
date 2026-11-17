"""Pay-as-you-go versus provisioned throughput (PTU) cost model.

Three deployment options are compared for each model class:

* ``payg``: every token is billed at the pay-as-you-go price.
* ``ptu``: enough PTUs to serve the peak minute, no pay-as-you-go spend.
* ``hybrid``: a smaller PTU base load with pay-as-you-go spillover.

The hybrid model works on hourly average demand. Bursts inside an hour spill
more traffic than estimated, so treat the hybrid result as a lower bound.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from archkit.capacity import ModelDemand
from archkit.pricing import ModelPrice, PtuSpec

Commitment = Literal["hourly", "monthly"]


@dataclass(frozen=True)
class DeploymentOption:
    """Monthly cost of one way to deploy a model class."""

    strategy: str
    ptus: int
    ptu_cost: float
    payg_cost: float
    spillover_fraction: float
    average_utilization: float

    @property
    def total(self) -> float:
        return self.ptu_cost + self.payg_cost


@dataclass(frozen=True)
class BreakEven:
    """Break-even figures between pay-as-you-go and provisioned throughput."""

    payg_cost_per_1m_weighted_tokens: float
    utilization: float
    monthly_requests: float | None

    @property
    def reachable(self) -> bool:
        """False when a fully used PTU is still more expensive than pay-as-you-go."""
        return self.utilization <= 1.0


def payg_monthly_cost(demand: ModelDemand, price: ModelPrice) -> float:
    """Monthly pay-as-you-go cost for the full demand of one model class."""
    return (
        demand.monthly_uncached_input_tokens * price.input_per_1m
        + demand.monthly_cached_input_tokens * price.cached_price
        + demand.monthly_output_tokens * price.output_per_1m
    ) / 1e6


def ptus_for_tpm(weighted_tpm: float, spec: PtuSpec) -> int:
    """PTUs needed for an input-equivalent TPM, rounded to the purchasable size."""
    if weighted_tpm <= 0:
        return 0
    raw = math.ceil(weighted_tpm / spec.input_tpm_per_ptu - 1e-9)
    if raw <= spec.min_ptus:
        return spec.min_ptus
    steps = math.ceil((raw - spec.min_ptus) / spec.increment)
    return spec.min_ptus + steps * spec.increment


def ptu_monthly_cost(ptus: int, spec: PtuSpec, hours_per_month: float, commitment: Commitment = "monthly") -> float:
    """Monthly cost of a PTU deployment with or without a reservation."""
    if commitment == "monthly":
        return ptus * spec.monthly_reservation_per_ptu
    return ptus * spec.hourly_per_ptu * hours_per_month


def hybrid_option(
    demand: ModelDemand,
    price: ModelPrice,
    ptus: int,
    hours_per_month: float,
    commitment: Commitment = "monthly",
    strategy: str = "hybrid",
) -> DeploymentOption:
    """Cost of ``ptus`` provisioned units with pay-as-you-go spillover above capacity."""
    if price.ptu is None:
        raise ValueError(f"model class '{demand.model}' has no PTU specification")
    spec = price.ptu
    capacity_tpm = ptus * spec.input_tpm_per_ptu
    hourly = demand.weighted_hourly_tpm(spec.output_token_weight)
    total = sum(hourly)
    spilled = sum(max(0.0, tpm - capacity_tpm) for tpm in hourly)
    spill_fraction = spilled / total if total > 0 else 0.0
    served = total - spilled
    utilization = served / (capacity_tpm * 24) if capacity_tpm > 0 else 0.0
    return DeploymentOption(
        strategy=strategy,
        ptus=ptus,
        ptu_cost=ptu_monthly_cost(ptus, spec, hours_per_month, commitment),
        payg_cost=payg_monthly_cost(demand, price) * spill_fraction,
        spillover_fraction=spill_fraction,
        average_utilization=utilization,
    )


def deployment_options(
    demand: ModelDemand, price: ModelPrice, hours_per_month: float, commitment: Commitment = "monthly"
) -> list[DeploymentOption]:
    """Return the pay-as-you-go, full PTU and best hybrid options, cheapest first."""
    payg = DeploymentOption("payg", 0, 0.0, payg_monthly_cost(demand, price), 1.0, 0.0)
    if price.ptu is None:
        return [payg]
    spec = price.ptu
    full_ptus = ptus_for_tpm(demand.weighted_peak_tpm(spec.output_token_weight), spec)
    full = hybrid_option(demand, price, full_ptus, hours_per_month, commitment, strategy="ptu")
    full = DeploymentOption("ptu", full.ptus, full.ptu_cost, 0.0, 0.0, full.average_utilization)
    options = [payg, full]
    candidates = range(spec.min_ptus, full_ptus, spec.increment)
    hybrids = [hybrid_option(demand, price, n, hours_per_month, commitment) for n in candidates]
    if hybrids:
        options.append(min(hybrids, key=lambda option: option.total))
    return sorted(options, key=lambda option: option.total)


def break_even(
    demand: ModelDemand, price: ModelPrice, hours_per_month: float, commitment: Commitment = "monthly"
) -> BreakEven | None:
    """Compute where a full PTU deployment becomes cheaper than pay-as-you-go.

    ``utilization`` is the average utilization at which one PTU costs the same as
    the pay-as-you-go tokens it serves, for this workload's input/output mix.
    ``monthly_requests`` is the smallest volume with the same traffic shape at
    which a peak-sized PTU deployment is no more expensive than pay-as-you-go.
    """
    if price.ptu is None:
        return None
    spec = price.ptu
    weighted_tokens = demand.monthly_input_tokens + spec.output_token_weight * demand.monthly_output_tokens
    if weighted_tokens <= 0:
        return None
    payg_per_weighted_token = payg_monthly_cost(demand, price) / weighted_tokens
    minutes = hours_per_month * 60
    full_use_value = spec.input_tpm_per_ptu * minutes * payg_per_weighted_token
    utilization = ptu_monthly_cost(1, spec, hours_per_month, commitment) / full_use_value

    volume = None
    if demand.monthly_requests > 0:
        for step in range(-200, 401):
            factor = 10 ** (step / 100)
            scaled = demand.scaled(factor)
            ptus = ptus_for_tpm(scaled.weighted_peak_tpm(spec.output_token_weight), spec)
            if ptu_monthly_cost(ptus, spec, hours_per_month, commitment) <= payg_monthly_cost(scaled, price):
                volume = scaled.monthly_requests
                break
    return BreakEven(payg_per_weighted_token * 1e6, utilization, volume)
