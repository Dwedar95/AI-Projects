"""Weighted multi-criteria scoring, ranking sensitivity and the build / buy / partner helper."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .models import BuildBuyInputs, Dimension, Portfolio, ScoringConfig, UseCase

DIMENSIONS: tuple[Dimension, ...] = ("value", "feasibility", "risk")


@dataclass(frozen=True)
class ScoredUseCase:
    """Scoring result for one use case. Dimension scores are normalised to 0..1."""

    id: str
    name: str
    value: float
    feasibility: float
    risk: float
    overall: float
    quadrant: str
    rank: int = 0


def dimension_scores(uc: UseCase, config: ScoringConfig) -> dict[Dimension, float]:
    """Weighted mean of the criteria of each dimension, normalised to 0..1."""
    span = config.scale_max - config.scale_min
    out: dict[Dimension, float] = {}
    for dim in DIMENSIONS:
        crits = [c for c in config.criteria if c.dimension == dim]
        total = sum(c.weight for c in crits)
        raw = sum(c.weight * uc.scores[c.name] for c in crits) / total
        out[dim] = (raw - config.scale_min) / span
    return out


def overall_score(dims: dict[Dimension, float], weights: dict[Dimension, float]) -> float:
    """Combine dimension scores into 0..100. Risk counts inverted (low risk is good)."""
    total = sum(weights[d] for d in DIMENSIONS)
    s = weights["value"] * dims["value"] + weights["feasibility"] * dims["feasibility"]
    s += weights["risk"] * (1.0 - dims["risk"])
    return 100.0 * s / total


def quadrant(value: float, feasibility: float, threshold: float = 0.5) -> str:
    """Classic value / feasibility quadrant label."""
    if value >= threshold and feasibility >= threshold:
        return "quick win"
    if value >= threshold:
        return "strategic bet"
    if feasibility >= threshold:
        return "fill-in"
    return "deprioritise"


def score_portfolio(portfolio: Portfolio, weights: dict[Dimension, float] | None = None) -> list[ScoredUseCase]:
    """Score and rank all use cases. Ties are broken by value, then by id for determinism."""
    w = weights or portfolio.scoring.dimension_weights
    rows = []
    for uc in portfolio.use_cases:
        d = dimension_scores(uc, portfolio.scoring)
        rows.append(
            ScoredUseCase(
                id=uc.id,
                name=uc.name,
                value=d["value"],
                feasibility=d["feasibility"],
                risk=d["risk"],
                overall=overall_score(d, w),
                quadrant=quadrant(d["value"], d["feasibility"]),
            )
        )
    rows.sort(key=lambda r: (-round(r.overall, 9), -r.value, r.id))
    return [ScoredUseCase(**{**r.__dict__, "rank": i + 1}) for i, r in enumerate(rows)]


def kendall_tau(order_a: list[str], order_b: list[str]) -> float:
    """Kendall rank correlation between two orderings of the same ids (1.0 means identical)."""
    if set(order_a) != set(order_b):
        raise ValueError("orderings must contain the same ids")
    n = len(order_a)
    if n < 2:
        return 1.0
    pos_b = {k: i for i, k in enumerate(order_b)}
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if pos_b[order_a[i]] < pos_b[order_a[j]]:
                concordant += 1
            else:
                discordant += 1
    return (concordant - discordant) / (n * (n - 1) / 2)


@dataclass
class OatResult:
    """Effect of shifting one dimension weight while rescaling the others."""

    dimension: str
    delta: float
    weights: dict[str, float]
    ranking: list[str]
    kendall_tau: float
    top_changed: bool
    max_rank_shift: int


@dataclass
class RankStability:
    """Rank distribution of one use case under randomly perturbed weights."""

    id: str
    name: str
    baseline_rank: int
    mean_rank: float
    best_rank: int
    worst_rank: int
    p_top_k: float
    p_same_rank: float


@dataclass
class SensitivityReport:
    """Output of :func:`sensitivity_analysis`."""

    baseline: list[ScoredUseCase]
    one_at_a_time: list[OatResult]
    stability: list[RankStability]
    mean_kendall_tau: float
    p_top1_unchanged: float
    n_samples: int
    seed: int
    top_k: int
    concentration: float
    switch_points: list[dict[str, object]] = field(default_factory=list)


def _shift_weight(weights: dict[Dimension, float], dim: Dimension, delta: float) -> dict[Dimension, float]:
    """Move ``dim`` by ``delta`` (absolute, on normalised weights) and rescale the rest to sum to 1."""
    total = sum(weights.values())
    norm = {d: weights[d] / total for d in DIMENSIONS}
    new = min(1.0, max(0.0, norm[dim] + delta))
    rest = 1.0 - norm[dim]
    out = {dim: new}
    for d in DIMENSIONS:
        if d != dim:
            out[d] = (norm[d] / rest) * (1.0 - new) if rest > 0 else (1.0 - new) / 2
    return out


def _switch_points(portfolio: Portfolio, baseline: list[ScoredUseCase], step: float = 0.005) -> list[dict[str, object]]:
    """Smallest single-weight shift (either direction) that changes the rank-1 use case."""
    out: list[dict[str, object]] = []
    base_w = portfolio.scoring.dimension_weights
    for dim in DIMENSIONS:
        found: dict[str, object] = {"dimension": dim, "delta": None, "new_top": None}
        k = 1
        while k * step <= 1.0 and found["delta"] is None:
            for sign in (1, -1):
                delta = sign * k * step
                ranked = score_portfolio(portfolio, _shift_weight(base_w, dim, delta))
                if ranked[0].id != baseline[0].id:
                    found = {"dimension": dim, "delta": round(delta, 4), "new_top": ranked[0].id}
                    break
            k += 1
        out.append(found)
    return out


def sensitivity_analysis(
    portfolio: Portfolio,
    deltas: tuple[float, ...] = (-0.10, 0.10),
    n_samples: int = 2000,
    concentration: float = 50.0,
    top_k: int = 3,
    seed: int = 42,
) -> SensitivityReport:
    """Quantify how robust the ranking is to the dimension weights.

    Two views are produced. One-at-a-time: each dimension weight is shifted by
    each value in ``deltas`` and the others are rescaled. Monte Carlo: weights
    are drawn from a Dirichlet distribution whose mean is the baseline weights;
    ``concentration`` controls the spread (higher means closer to baseline).
    """
    baseline = score_portfolio(portfolio)
    base_order = [r.id for r in baseline]
    base_rank = {r.id: r.rank for r in baseline}
    base_w = portfolio.scoring.dimension_weights

    oat: list[OatResult] = []
    for dim in DIMENSIONS:
        for delta in deltas:
            w = _shift_weight(base_w, dim, delta)
            ranked = score_portfolio(portfolio, w)
            order = [r.id for r in ranked]
            shift = max(abs(r.rank - base_rank[r.id]) for r in ranked)
            oat.append(
                OatResult(
                    dimension=dim,
                    delta=delta,
                    weights={d: round(w[d], 4) for d in DIMENSIONS},
                    ranking=order,
                    kendall_tau=kendall_tau(base_order, order),
                    top_changed=order[0] != base_order[0],
                    max_rank_shift=shift,
                )
            )

    rng = np.random.default_rng(seed)
    total = sum(base_w.values())
    alpha = np.array([base_w[d] / total for d in DIMENSIONS]) * concentration
    alpha = np.maximum(alpha, 1e-6)
    samples = rng.dirichlet(alpha, size=n_samples)
    # Vectorised scoring: one row per use case, one column per dimension contribution.
    dims = np.array(
        [[r.value, r.feasibility, 1.0 - r.risk] for r in sorted(baseline, key=lambda r: r.id)]
    )
    ids = sorted(base_order)
    scores = samples @ dims.T  # (n_samples, n_use_cases)
    order_idx = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty_like(order_idx)
    rows = np.arange(n_samples)[:, None]
    ranks[rows, order_idx] = np.arange(1, len(ids) + 1)[None, :]

    by_id = {r.id: r for r in baseline}
    stability = []
    for j, uid in enumerate(ids):
        col = ranks[:, j]
        stability.append(
            RankStability(
                id=uid,
                name=by_id[uid].name,
                baseline_rank=base_rank[uid],
                mean_rank=float(col.mean()),
                best_rank=int(col.min()),
                worst_rank=int(col.max()),
                p_top_k=float((col <= top_k).mean()),
                p_same_rank=float((col == base_rank[uid]).mean()),
            )
        )
    stability.sort(key=lambda s: s.baseline_rank)
    taus = [kendall_tau(base_order, [ids[i] for i in row]) for row in order_idx]
    top_idx = ids.index(base_order[0])
    return SensitivityReport(
        baseline=baseline,
        one_at_a_time=oat,
        stability=stability,
        mean_kendall_tau=float(np.mean(taus)),
        p_top1_unchanged=float((order_idx[:, 0] == top_idx).mean()),
        n_samples=n_samples,
        seed=seed,
        top_k=top_k,
        concentration=concentration,
        switch_points=_switch_points(portfolio, baseline),
    )


# Contribution of each factor (centred to -1..1) to each sourcing option.
# Positive means the factor being high argues for the option.
BUILD_BUY_MATRIX: dict[str, dict[str, float]] = {
    "build": {
        "strategic_differentiation": 1.0,
        "internal_capability": 1.0,
        "market_solution_maturity": -0.8,
        "data_sensitivity": 0.6,
        "time_pressure": -0.8,
        "customisation_need": 0.8,
    },
    "buy": {
        "strategic_differentiation": -1.0,
        "internal_capability": -0.4,
        "market_solution_maturity": 1.0,
        "data_sensitivity": -0.5,
        "time_pressure": 0.8,
        "customisation_need": -0.8,
    },
    "partner": {
        "strategic_differentiation": 0.4,
        "internal_capability": -1.0,
        "market_solution_maturity": 0.1,
        "data_sensitivity": -0.1,
        "time_pressure": 0.3,
        "customisation_need": 0.5,
    },
}


@dataclass
class BuildBuyDecision:
    """Recommendation with per-option scores and the factors that drove it."""

    recommendation: str
    scores: dict[str, float]
    margin: float
    rationale: list[str]


def build_buy_partner(inputs: BuildBuyInputs) -> BuildBuyDecision:
    """Score build, buy and partner with a transparent linear factor matrix.

    Each 1..5 rating is centred to -1..1 and multiplied by the option's factor
    coefficient. The margin between the best and second option indicates how
    clear the call is; a margin below 0.5 is reported as a close call.
    """
    centred = {k: (v - 3) / 2 for k, v in inputs.model_dump().items()}
    scores = {
        opt: round(sum(coef * centred[f] for f, coef in coefs.items()), 3)
        for opt, coefs in BUILD_BUY_MATRIX.items()
    }
    ordered = sorted(scores, key=lambda o: (-scores[o], o))
    best = ordered[0]
    margin = round(scores[best] - scores[ordered[1]], 3)
    contributions = sorted(
        ((BUILD_BUY_MATRIX[best][f] * centred[f], f) for f in centred), reverse=True
    )
    rationale = [
        f"{f.replace('_', ' ')} rated {getattr(inputs, f)}/5 supports {best} ({c:+.2f})"
        for c, f in contributions[:3]
        if c > 0
    ]
    against = [f for c, f in contributions if c < 0]
    if against:
        rationale.append("factors against: " + ", ".join(f.replace("_", " ") for f in against))
    if margin < 0.5:
        rationale.append(f"close call: {ordered[1]} is within {margin:.2f} points, validate with a short spike")
    return BuildBuyDecision(recommendation=best, scores=scores, margin=margin, rationale=rationale)
