"""Business-case model: cash flows, NPV, payback, IRR, tornado and Monte Carlo uncertainty."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import Economics

DRIVERS = ("benefit", "build_cost", "run_cost", "volume", "adoption")


@dataclass
class CashFlowRow:
    """Cash flow of one year. Year 0 is the investment year with no benefits."""

    year: int
    adoption: float
    benefit: float
    build_cost: float
    change_cost: float
    run_cost: float
    inference_cost: float

    @property
    def net(self) -> float:
        return self.benefit - self.build_cost - self.change_cost - self.run_cost - self.inference_cost


@dataclass
class BusinessCase:
    """Deterministic business case built from the base values."""

    rows: list[CashFlowRow]
    npv: float
    total_cost: float
    total_benefit: float
    roi: float
    payback_years: float | None
    irr: float | None


def inference_cost_per_year(e: Economics, volume_mult: float = 1.0) -> float:
    """Token cost at full adoption: requests times (input and output tokens priced per 1k)."""
    per_request = (
        e.input_tokens_per_request / 1000.0 * e.price_per_1k_input_tokens
        + e.output_tokens_per_request / 1000.0 * e.price_per_1k_output_tokens
    )
    return e.requests_per_year * volume_mult * per_request


def cash_flows(e: Economics, mult: dict[str, float] | None = None) -> list[CashFlowRow]:
    """Yearly cash flows. ``mult`` scales the drivers listed in :data:`DRIVERS`.

    Benefits and inference cost scale with adoption. Adoption multipliers are
    capped so that adoption never exceeds 100 percent.
    """
    m = {d: 1.0 for d in DRIVERS}
    m.update(mult or {})
    rows = [CashFlowRow(0, 0.0, 0.0, e.build_cost * m["build_cost"], e.change_management_cost, 0.0, 0.0)]
    full_inference = inference_cost_per_year(e, m["volume"])
    for year in range(1, e.horizon_years + 1):
        adoption = min(1.0, e.adoption(year) * m["adoption"])
        rows.append(
            CashFlowRow(
                year=year,
                adoption=adoption,
                benefit=e.annual_benefit_at_full_adoption * m["benefit"] * adoption,
                build_cost=0.0,
                change_cost=0.0,
                run_cost=e.run_cost_per_year * m["run_cost"],
                inference_cost=full_inference * adoption,
            )
        )
    return rows


def npv(net_flows: list[float], rate: float) -> float:
    """Net present value with the first flow at time zero."""
    return float(sum(cf / (1.0 + rate) ** t for t, cf in enumerate(net_flows)))


def payback_period(net_flows: list[float]) -> float | None:
    """Undiscounted payback in years with linear interpolation inside the year, or None."""
    cumulative = 0.0
    for t, cf in enumerate(net_flows):
        previous = cumulative
        cumulative += cf
        if cumulative >= 0:
            if t == 0:
                return 0.0
            return (t - 1) + (-previous / cf if cf > 0 else 0.0)
    return None


def irr(net_flows: list[float], lo: float = -0.99, hi: float = 10.0, tol: float = 1e-7) -> float | None:
    """Internal rate of return by bisection. None when NPV does not change sign on the bracket."""
    f_lo, f_hi = npv(net_flows, lo), npv(net_flows, hi)
    if f_lo * f_hi > 0:
        return None
    for _ in range(200):
        mid = (lo + hi) / 2
        f_mid = npv(net_flows, mid)
        if abs(f_mid) < tol or (hi - lo) < tol:
            return mid
        if f_lo * f_mid <= 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2


def business_case(e: Economics, mult: dict[str, float] | None = None) -> BusinessCase:
    """Build the deterministic business case."""
    rows = cash_flows(e, mult)
    net = [r.net for r in rows]
    total_benefit = sum(r.benefit for r in rows)
    total_cost = total_benefit - sum(net)
    return BusinessCase(
        rows=rows,
        npv=npv(net, e.discount_rate),
        total_cost=total_cost,
        total_benefit=total_benefit,
        roi=(total_benefit - total_cost) / total_cost if total_cost > 0 else float("inf"),
        payback_years=payback_period(net),
        irr=irr(net),
    )


@dataclass
class TornadoBar:
    """NPV when one driver is set to its low and high multiplier, others at base."""

    driver: str
    npv_low: float
    npv_high: float

    @property
    def swing(self) -> float:
        return abs(self.npv_high - self.npv_low)


def tornado(e: Economics) -> list[TornadoBar]:
    """One-at-a-time NPV sensitivity for every driver with a declared range, largest swing first."""
    bars = []
    for driver, rng in e.uncertainty.items():
        low = business_case(e, {driver: rng.low}).npv
        high = business_case(e, {driver: rng.high}).npv
        bars.append(TornadoBar(driver, low, high))
    return sorted(bars, key=lambda b: -b.swing)


@dataclass
class MonteCarloResult:
    """Distribution summary of the simulated NPV."""

    n: int
    seed: int
    npv_mean: float
    npv_p10: float
    npv_p50: float
    npv_p90: float
    prob_negative_npv: float
    payback_p50: float | None
    prob_no_payback: float


def monte_carlo(e: Economics, n: int = 5000, seed: int = 42) -> MonteCarloResult:
    """Simulate NPV with independent triangular multipliers (low, 1.0, high) per driver.

    Drivers without a declared range stay at their base value. Independence of
    the drivers is a simplification and is stated in the README limitations.
    """
    rng = np.random.default_rng(seed)
    mults = {d: np.ones(n) for d in DRIVERS}
    for driver in DRIVERS:  # fixed iteration order keeps results reproducible
        if driver in e.uncertainty:
            r = e.uncertainty[driver]
            if r.high > r.low:
                mults[driver] = rng.triangular(r.low, 1.0, r.high, size=n)

    years = np.arange(1, e.horizon_years + 1)
    base_adoption = np.array([e.adoption(int(y)) for y in years])
    adoption = np.minimum(1.0, base_adoption[None, :] * mults["adoption"][:, None])
    benefit = e.annual_benefit_at_full_adoption * mults["benefit"][:, None] * adoption
    infer = inference_cost_per_year(e) * mults["volume"][:, None] * adoption
    run = e.run_cost_per_year * mults["run_cost"][:, None] * np.ones_like(adoption)
    flows = np.empty((n, e.horizon_years + 1))
    flows[:, 0] = -(e.build_cost * mults["build_cost"] + e.change_management_cost)
    flows[:, 1:] = benefit - infer - run
    discount = (1.0 + e.discount_rate) ** np.arange(e.horizon_years + 1)
    npvs = (flows / discount[None, :]).sum(axis=1)

    paybacks = np.full(n, np.nan)
    for i in range(n):
        pb = payback_period(list(flows[i]))
        if pb is not None:
            paybacks[i] = pb
    has_payback = ~np.isnan(paybacks)
    return MonteCarloResult(
        n=n,
        seed=seed,
        npv_mean=float(npvs.mean()),
        npv_p10=float(np.percentile(npvs, 10)),
        npv_p50=float(np.percentile(npvs, 50)),
        npv_p90=float(np.percentile(npvs, 90)),
        prob_negative_npv=float((npvs < 0).mean()),
        payback_p50=float(np.median(paybacks[has_payback])) if has_payback.any() else None,
        prob_no_payback=float((~has_payback).mean()),
    )
