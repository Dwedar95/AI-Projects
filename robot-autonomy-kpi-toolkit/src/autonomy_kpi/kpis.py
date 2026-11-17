"""KPI computation from mission logs, with mission-level bootstrap intervals.

All fleet KPIs are ratios of sums over missions (for example total
interventions divided by total hours). Confidence intervals come from a
cluster bootstrap that resamples whole missions, because events inside one
mission are not independent.

The definitions here are the single source of truth. ``docs/kpi_glossary.md``
describes them in words and the tests check that both agree on the names.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from .events import Event, group_by_mission


@dataclass(frozen=True)
class MissionSummary:
    """Per-mission aggregates from which all KPIs are computed."""

    mission_id: str
    build: str
    platform: str
    environment: str
    terrain: str
    outcome: str
    duration_s: float
    distance_m: float
    autonomous_s: float
    interventions: int
    proposed: int
    withheld: int
    withheld_by_gate: int
    min_separation_m: float
    loc_error_sum_m: float
    loc_error_samples: int
    faults: int

    @property
    def success(self) -> bool:
        return self.outcome == "success"


def summarise_mission(events: list[Event]) -> MissionSummary:
    """Reduce the time-ordered events of one mission to a :class:`MissionSummary`.

    Autonomous time is the time-weighted share of ``state`` samples in mode
    ``autonomous``: each sample accounts for the interval since the previous
    sample (or since mission start).
    """
    start = next(e for e in events if e.type == "mission_start")
    end = next(e for e in events if e.type == "mission_end")
    states = [e for e in events if e.type == "state"]
    autonomous_s = 0.0
    prev_t = 0.0
    for s in states:
        if s.get("mode") == "autonomous":
            autonomous_s += s.t - prev_t
        prev_t = s.t
    withheld = [e for e in events if e.type == "command_withheld"]
    return MissionSummary(
        mission_id=start.mission_id,
        build=start.build,
        platform=start.get("platform"),
        environment=start.get("environment"),
        terrain=start.get("terrain"),
        outcome=end.get("outcome"),
        duration_s=float(end.get("duration_s")),
        distance_m=float(end.get("distance_m")),
        autonomous_s=autonomous_s,
        interventions=sum(1 for e in events if e.type == "intervention"),
        proposed=sum(1 for e in events if e.type == "command_proposed"),
        withheld=len(withheld),
        withheld_by_gate=sum(1 for e in withheld if e.get("reason") != "manual_control"),
        min_separation_m=min((s.get("separation_m") for s in states), default=math.inf),
        loc_error_sum_m=float(sum(s.get("localisation_error_m") for s in states)),
        loc_error_samples=len(states),
        faults=sum(1 for e in events if e.type == "fault"),
    )


def summarise_log(events: Iterable[Event]) -> list[MissionSummary]:
    """Summaries of all missions in a log, ordered by mission id."""
    groups = group_by_mission(events)
    return [summarise_mission(groups[mid]) for mid in sorted(groups)]


def _ratio(num: float, den: float) -> float:
    return num / den if den > 0 else math.nan


def _success_rate(m: list[MissionSummary]) -> float:
    return _ratio(sum(s.success for s in m), len(m))


def _interventions_per_hour(m: list[MissionSummary]) -> float:
    return _ratio(sum(s.interventions for s in m), sum(s.duration_s for s in m) / 3600.0)


def _interventions_per_km(m: list[MissionSummary]) -> float:
    return _ratio(sum(s.interventions for s in m), sum(s.distance_m for s in m) / 1000.0)


def _mdbi_km(m: list[MissionSummary]) -> float:
    n = sum(s.interventions for s in m)
    return sum(s.distance_m for s in m) / 1000.0 / n if n > 0 else math.inf


def _withhold_rate(m: list[MissionSummary]) -> float:
    return _ratio(sum(s.withheld_by_gate for s in m), sum(s.proposed for s in m))


def _min_separation(m: list[MissionSummary]) -> float:
    return min((s.min_separation_m for s in m), default=math.nan)


def _uptime(m: list[MissionSummary]) -> float:
    return _ratio(sum(s.autonomous_s for s in m), sum(s.duration_s for s in m))


def _loc_error(m: list[MissionSummary]) -> float:
    return _ratio(sum(s.loc_error_sum_m for s in m), sum(s.loc_error_samples for s in m))


def _faults_per_hour(m: list[MissionSummary]) -> float:
    return _ratio(sum(s.faults for s in m), sum(s.duration_s for s in m) / 3600.0)


@dataclass(frozen=True)
class KpiDefinition:
    """Name, unit, direction and estimator of a KPI."""

    name: str
    unit: str
    higher_is_better: bool
    description: str
    compute: Callable[[list[MissionSummary]], float]


KPI_DEFINITIONS: tuple[KpiDefinition, ...] = (
    KpiDefinition(
        "mission_success_rate",
        "fraction",
        True,
        "Missions with outcome success divided by all missions.",
        _success_rate,
    ),
    KpiDefinition(
        "interventions_per_hour",
        "1/h",
        False,
        "Intervention events divided by total mission hours.",
        _interventions_per_hour,
    ),
    KpiDefinition(
        "interventions_per_km",
        "1/km",
        False,
        "Intervention events divided by total distance in km.",
        _interventions_per_km,
    ),
    KpiDefinition(
        "mean_distance_between_interventions_km",
        "km",
        True,
        "Total distance in km divided by intervention events (infinite if none).",
        _mdbi_km,
    ),
    KpiDefinition(
        "gate_withhold_rate",
        "fraction",
        False,
        "Commands withheld by the safety gate (manual-control withholds excluded) divided by "
        "proposed commands.",
        _withhold_rate,
    ),
    KpiDefinition(
        "min_separation_m",
        "m",
        True,
        "Smallest separation distance reported in any state sample.",
        _min_separation,
    ),
    KpiDefinition(
        "autonomy_uptime",
        "fraction",
        True,
        "Time in autonomous mode divided by total mission time.",
        _uptime,
    ),
    KpiDefinition(
        "localisation_error_mean_m",
        "m",
        False,
        "Mean localisation error over all state samples.",
        _loc_error,
    ),
    KpiDefinition(
        "faults_per_hour",
        "1/h",
        False,
        "Fault events divided by total mission hours.",
        _faults_per_hour,
    ),
)

KPI_NAMES: tuple[str, ...] = tuple(d.name for d in KPI_DEFINITIONS)
KPI_BY_NAME: dict[str, KpiDefinition] = {d.name: d for d in KPI_DEFINITIONS}


@dataclass(frozen=True)
class KpiValue:
    """Point estimate with a bootstrap confidence interval."""

    name: str
    value: float
    ci_low: float
    ci_high: float
    unit: str
    higher_is_better: bool
    n_missions: int


def bootstrap_ci(
    missions: list[MissionSummary],
    statistic: Callable[[list[MissionSummary]], float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval, resampling missions with replacement.

    Non-finite replicates (for example an infinite distance between
    interventions when a resample contains none) are kept: an upper bound of
    ``inf`` is the honest answer in that case. NaN replicates are dropped.
    """
    n = len(missions)
    if n < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    reps = np.array([statistic([missions[i] for i in row]) for row in idx], dtype=float)
    reps = np.sort(reps[~np.isnan(reps)])
    if len(reps) == 0:
        return math.nan, math.nan
    alpha = (1.0 - confidence) / 2.0
    lo = reps[int(np.floor(alpha * (len(reps) - 1)))]
    hi = reps[int(np.ceil((1.0 - alpha) * (len(reps) - 1)))]
    return float(lo), float(hi)


def compute_kpis(
    missions: list[MissionSummary],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, KpiValue]:
    """Compute every KPI with its bootstrap interval."""
    if not missions:
        raise ValueError("no missions to compute KPIs from")
    out: dict[str, KpiValue] = {}
    for d in KPI_DEFINITIONS:
        lo, hi = bootstrap_ci(missions, d.compute, n_resamples, confidence, seed)
        out[d.name] = KpiValue(
            d.name, d.compute(missions), lo, hi, d.unit, d.higher_is_better, len(missions)
        )
    return out
