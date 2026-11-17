"""Release regression gates: candidate build against a baseline build.

For each gated KPI the candidate is compared with the baseline in two ways:

1. Effect size: the relative regression (change in the unfavourable
   direction divided by the baseline value) against a configurable tolerance.
2. Evidence: a one-sided statistical test of "the candidate is worse".

   * ``fisher``: Fisher's exact test on mission outcome counts.
   * ``poisson_rate``: exact conditional test for two Poisson rates (the
     candidate's share of all events is binomial under equal rates).
   * ``mann_whitney``: Mann-Whitney U test on per-mission values, used for
     KPIs whose samples inside a mission are correlated.

Verdict per KPI:

* FAIL: an absolute limit (``min_value`` / ``max_value``) is violated, or the
  regression exceeds the tolerance and the test is significant at ``alpha``.
* WARN: the regression exceeds the tolerance without statistical support
  (often too few missions), or it is significant but within the tolerance.
* PASS: otherwise.

No correction for multiple comparisons is applied. With several gates the
chance of at least one false alarm is higher than ``alpha``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy import stats

from .kpis import KPI_BY_NAME, KPI_NAMES, MissionSummary

Verdict = Literal["PASS", "WARN", "FAIL"]


class GateThreshold(BaseModel):
    """Thresholds of one KPI gate."""

    model_config = ConfigDict(extra="forbid")

    max_relative_regression: float = Field(default=0.10, ge=0)
    min_value: float | None = None
    max_value: float | None = None


class GateConfig(BaseModel):
    """Gate configuration loaded from YAML."""

    model_config = ConfigDict(extra="forbid")

    alpha: float = Field(default=0.05, gt=0, lt=1)
    gates: dict[str, GateThreshold]

    @model_validator(mode="after")
    def _known_kpis(self) -> GateConfig:
        unknown = sorted(set(self.gates) - set(KPI_NAMES))
        if unknown:
            raise ValueError(f"gates refer to unknown KPIs: {unknown}")
        return self


def load_gate_config(path: str | Path) -> GateConfig:
    """Load a gate configuration from a YAML file."""
    with Path(path).open(encoding="utf-8") as fh:
        return GateConfig.model_validate(yaml.safe_load(fh))


def fisher_worse(base: list[MissionSummary], cand: list[MissionSummary]) -> float:
    """One-sided p-value that the candidate success probability is lower."""
    table = [
        [sum(m.success for m in cand), sum(not m.success for m in cand)],
        [sum(m.success for m in base), sum(not m.success for m in base)],
    ]
    return float(stats.fisher_exact(table, alternative="less")[1])


def poisson_rate_worse(
    count_base: int, exposure_base: float, count_cand: int, exposure_cand: float
) -> float:
    """One-sided p-value that the candidate event rate is higher.

    Conditional on the total count, the candidate count is binomial with
    probability ``exposure_cand / (exposure_cand + exposure_base)`` when both
    builds have the same rate.
    """
    total = count_base + count_cand
    if total == 0 or exposure_base <= 0 or exposure_cand <= 0:
        return 1.0
    p0 = exposure_cand / (exposure_cand + exposure_base)
    return float(stats.binomtest(count_cand, total, p0, alternative="greater").pvalue)


def mann_whitney_worse(base: list[float], cand: list[float], higher_is_better: bool) -> float:
    """One-sided p-value that candidate per-mission values are worse."""
    base = [v for v in base if math.isfinite(v)]
    cand = [v for v in cand if math.isfinite(v)]
    if len(base) < 2 or len(cand) < 2 or len(set(base) | set(cand)) < 2:
        return 1.0
    alternative = "less" if higher_is_better else "greater"
    return float(stats.mannwhitneyu(cand, base, alternative=alternative).pvalue)


def _events_per(
    count: Callable[[MissionSummary], float], exposure: Callable[[MissionSummary], float]
):
    def test(base: list[MissionSummary], cand: list[MissionSummary]) -> float:
        return poisson_rate_worse(
            int(sum(count(m) for m in base)),
            sum(exposure(m) for m in base),
            int(sum(count(m) for m in cand)),
            sum(exposure(m) for m in cand),
        )

    return test


def _per_mission(value: Callable[[MissionSummary], float], higher_is_better: bool):
    def test(base: list[MissionSummary], cand: list[MissionSummary]) -> float:
        return mann_whitney_worse(
            [value(m) for m in base], [value(m) for m in cand], higher_is_better
        )

    return test


def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else math.nan


# KPI name -> (test name, function returning the one-sided p-value "candidate worse").
KPI_TESTS: dict[str, tuple[str, Callable[[list[MissionSummary], list[MissionSummary]], float]]] = {
    "mission_success_rate": ("fisher", fisher_worse),
    "interventions_per_hour": (
        "poisson_rate",
        _events_per(lambda m: m.interventions, lambda m: m.duration_s),
    ),
    "interventions_per_km": (
        "poisson_rate",
        _events_per(lambda m: m.interventions, lambda m: m.distance_m),
    ),
    "mean_distance_between_interventions_km": (
        "poisson_rate",
        _events_per(lambda m: m.interventions, lambda m: m.distance_m),
    ),
    "faults_per_hour": ("poisson_rate", _events_per(lambda m: m.faults, lambda m: m.duration_s)),
    "gate_withhold_rate": (
        "mann_whitney",
        _per_mission(lambda m: _safe_div(m.withheld_by_gate, m.proposed), False),
    ),
    "autonomy_uptime": (
        "mann_whitney",
        _per_mission(lambda m: _safe_div(m.autonomous_s, m.duration_s), True),
    ),
    "localisation_error_mean_m": (
        "mann_whitney",
        _per_mission(lambda m: _safe_div(m.loc_error_sum_m, m.loc_error_samples), False),
    ),
    "min_separation_m": ("mann_whitney", _per_mission(lambda m: m.min_separation_m, True)),
}


@dataclass(frozen=True)
class GateResult:
    """Outcome of one KPI gate."""

    kpi: str
    baseline: float
    candidate: float
    relative_regression: float
    test: str
    p_value: float
    verdict: Verdict
    reason: str


def relative_regression(baseline: float, candidate: float, higher_is_better: bool) -> float:
    """Change in the unfavourable direction relative to the baseline (positive is worse)."""
    if math.isinf(baseline) and math.isinf(candidate):
        return 0.0
    if math.isinf(baseline):
        return math.inf
    if math.isinf(candidate):
        return -math.inf
    delta = (baseline - candidate) if higher_is_better else (candidate - baseline)
    if baseline == 0:
        return 0.0 if delta == 0 else math.copysign(math.inf, delta)
    return delta / abs(baseline)


def evaluate_gates(
    baseline: list[MissionSummary], candidate: list[MissionSummary], config: GateConfig
) -> list[GateResult]:
    """Evaluate every configured gate."""
    if not baseline or not candidate:
        raise ValueError("both baseline and candidate need at least one mission")
    results = []
    for name, threshold in config.gates.items():
        definition = KPI_BY_NAME[name]
        base_value = definition.compute(baseline)
        cand_value = definition.compute(candidate)
        regression = relative_regression(base_value, cand_value, definition.higher_is_better)
        test_name, test_fn = KPI_TESTS[name]
        p_value = test_fn(baseline, candidate)
        significant = p_value < config.alpha
        exceeds = regression > threshold.max_relative_regression
        if threshold.min_value is not None and cand_value < threshold.min_value:
            verdict, reason = "FAIL", f"below absolute minimum {threshold.min_value}"
        elif threshold.max_value is not None and cand_value > threshold.max_value:
            verdict, reason = "FAIL", f"above absolute maximum {threshold.max_value}"
        elif exceeds and significant:
            verdict, reason = "FAIL", "regression beyond tolerance, statistically significant"
        elif exceeds:
            verdict, reason = "WARN", "regression beyond tolerance, not significant (more data)"
        elif significant and regression > 0:
            verdict, reason = "WARN", "significant regression within tolerance"
        else:
            verdict, reason = "PASS", "no regression beyond tolerance"
        results.append(
            GateResult(
                name, base_value, cand_value, regression, test_name, p_value, verdict, reason
            )
        )
    return results


def overall_verdict(results: list[GateResult]) -> Verdict:
    """FAIL if any gate fails, WARN if any warns, else PASS."""
    verdicts = {r.verdict for r in results}
    if "FAIL" in verdicts:
        return "FAIL"
    return "WARN" if "WARN" in verdicts else "PASS"
