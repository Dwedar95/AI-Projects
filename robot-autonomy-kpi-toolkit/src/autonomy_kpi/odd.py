"""Operational Design Domain (ODD): YAML schema, validator and coverage checks.

An ODD states the conditions under which autonomous operation is intended:
environment, terrain, lighting, speed and connectivity limits. This module

* validates an ODD document (:func:`validate_odd`),
* checks each mission log against it (:func:`check_mission`), reporting the
  share of autonomous time spent inside the ODD and the violations, and
* measures how much of the ODD the logged missions exercised
  (:func:`odd_coverage`), which shows where test evidence is missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .events import Event
from .synth import ENVIRONMENTS, PLATFORMS, TERRAINS

# Lighting bands used for coverage cells, as (name, lower lux, upper lux).
LIGHTING_BANDS: tuple[tuple[str, float, float], ...] = (
    ("dark", 0.0, 50.0),
    ("dim", 50.0, 500.0),
    ("bright_indoor", 500.0, 5000.0),
    ("daylight", 5000.0, float("inf")),
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EnvironmentSpec(_Strict):
    allowed: list[str] = Field(min_length=1)


class TerrainSpec(_Strict):
    allowed: list[str] = Field(min_length=1)
    max_slope_deg: float = Field(gt=0, le=60)


class LightingSpec(_Strict):
    min_lux: float = Field(ge=0)
    max_lux: float = Field(gt=0)


class SpeedSpec(_Strict):
    max_mps: float = Field(gt=0)


class ConnectivitySpec(_Strict):
    max_latency_ms: float = Field(gt=0)


class OddSpec(_Strict):
    """Operational Design Domain of one platform and software line."""

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    platforms: list[str] = Field(min_length=1)
    environment: EnvironmentSpec
    terrain: TerrainSpec
    lighting: LightingSpec
    speed: SpeedSpec
    connectivity: ConnectivitySpec

    @model_validator(mode="after")
    def _semantic_checks(self) -> OddSpec:
        problems = []
        for label, values, known in (
            ("platforms", self.platforms, PLATFORMS),
            ("environment.allowed", self.environment.allowed, ENVIRONMENTS),
            ("terrain.allowed", self.terrain.allowed, TERRAINS),
        ):
            unknown = sorted(set(values) - set(known))
            if unknown:
                problems.append(f"{label}: unknown values {unknown}, known: {list(known)}")
            if len(set(values)) != len(values):
                problems.append(f"{label}: duplicate values")
        if self.lighting.min_lux >= self.lighting.max_lux:
            problems.append("lighting: min_lux must be below max_lux")
        if problems:
            raise ValueError("; ".join(problems))
        return self

    def lighting_bands(self) -> list[str]:
        """Names of the lighting bands that overlap the allowed lux range."""
        return [
            name
            for name, lo, hi in LIGHTING_BANDS
            if lo < self.lighting.max_lux and hi > self.lighting.min_lux
        ]


def validate_odd(document: Any) -> tuple[OddSpec | None, list[str]]:
    """Validate a parsed ODD document. Returns ``(spec, problems)``.

    The document must be a mapping with a top-level ``odd`` key. On failure
    ``spec`` is None and ``problems`` lists every issue found.
    """
    if not isinstance(document, dict) or "odd" not in document:
        return None, ["document must be a mapping with a top-level 'odd' key"]
    try:
        return OddSpec.model_validate(document["odd"]), []
    except ValidationError as exc:
        problems = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"]) or "odd"
            problems.append(f"{loc}: {err['msg']}")
        return None, problems


def load_odd(path: str | Path) -> OddSpec:
    """Load and validate an ODD YAML file. Raises ``ValueError`` listing all problems."""
    with Path(path).open(encoding="utf-8") as fh:
        spec, problems = validate_odd(yaml.safe_load(fh))
    if spec is None:
        raise ValueError(f"invalid ODD {path}: " + "; ".join(problems))
    return spec


def lighting_band(lux: float) -> str:
    """Name of the lighting band a lux value falls into."""
    for name, lo, hi in LIGHTING_BANDS:
        if lo <= lux < hi:
            return name
    return LIGHTING_BANDS[-1][0]


@dataclass(frozen=True)
class OddViolation:
    """A contiguous span of autonomous operation outside one ODD limit."""

    mission_id: str
    dimension: str
    t_start: float
    t_end: float
    worst_value: float | str
    limit: str


@dataclass
class MissionOddResult:
    """ODD check of one mission."""

    mission_id: str
    autonomous_samples: int
    in_odd_samples: int
    cell: tuple[str, str, str]
    violations: list[OddViolation] = field(default_factory=list)

    @property
    def in_odd_fraction(self) -> float:
        """Share of autonomous state samples inside all ODD limits (1.0 if none)."""
        if self.autonomous_samples == 0:
            return 1.0
        return self.in_odd_samples / self.autonomous_samples


def check_mission(
    odd: OddSpec, events: list[Event], speed_tolerance: float = 0.05
) -> MissionOddResult:
    """Check the time-ordered events of one mission against the ODD.

    Declared conditions (platform, environment, terrain) are checked once.
    Dynamic limits (speed, lighting, latency, slope) are checked on every
    ``state`` sample in autonomous mode, since the ODD constrains autonomous
    operation only. Consecutive violating samples of one dimension are merged
    into one span.
    """
    start = next(e for e in events if e.type == "mission_start")
    states = [e for e in events if e.type == "state" and e.get("mode") == "autonomous"]
    violations: list[OddViolation] = []
    declared_ok = True
    for dim, value, allowed in (
        ("platform", start.get("platform"), odd.platforms),
        ("environment", start.get("environment"), odd.environment.allowed),
        ("terrain", start.get("terrain"), odd.terrain.allowed),
    ):
        if value not in allowed:
            declared_ok = False
            end_t = events[-1].t
            violations.append(
                OddViolation(start.mission_id, dim, 0.0, end_t, value, f"allowed: {allowed}")
            )

    limits = (
        (
            "speed",
            "speed_mps",
            lambda v: v > odd.speed.max_mps * (1 + speed_tolerance),
            f"<= {odd.speed.max_mps} m/s",
            max,
        ),
        (
            "lighting",
            "lighting_lux",
            lambda v: not odd.lighting.min_lux <= v <= odd.lighting.max_lux,
            f"{odd.lighting.min_lux} to {odd.lighting.max_lux} lux",
            None,
        ),
        (
            "connectivity",
            "link_latency_ms",
            lambda v: v > odd.connectivity.max_latency_ms,
            f"<= {odd.connectivity.max_latency_ms} ms",
            max,
        ),
        (
            "slope",
            "slope_deg",
            lambda v: v > odd.terrain.max_slope_deg,
            f"<= {odd.terrain.max_slope_deg} deg",
            max,
        ),
    )
    bad_sample = [False] * len(states)
    for dim, key, is_bad, limit_text, worst_fn in limits:
        span: list[Event] = []
        for i, s in enumerate([*states, None]):
            if s is not None and is_bad(s.get(key)):
                bad_sample[i] = True
                span.append(s)
                continue
            if span:
                values = [e.get(key) for e in span]
                if worst_fn is not None:
                    worst = worst_fn(values)
                else:  # two-sided limit: report the value furthest outside the range
                    mid = 0.5 * (odd.lighting.min_lux + odd.lighting.max_lux)
                    worst = max(values, key=lambda v: abs(v - mid))
                violations.append(
                    OddViolation(start.mission_id, dim, span[0].t, span[-1].t, worst, limit_text)
                )
                span = []
    in_odd = 0 if not declared_ok else sum(1 for bad in bad_sample if not bad)
    lux_values = sorted(e.get("lighting_lux") for e in events if e.type == "state")
    median_lux = lux_values[len(lux_values) // 2] if lux_values else 0.0
    cell = (start.get("environment"), start.get("terrain"), lighting_band(median_lux))
    return MissionOddResult(start.mission_id, len(states), in_odd, cell, violations)


@dataclass(frozen=True)
class OddCoverage:
    """How much of the ODD's condition space was exercised by a set of missions."""

    total_cells: int
    covered_cells: int
    missions_per_cell: dict[tuple[str, str, str], int]
    uncovered: list[tuple[str, str, str]]
    out_of_odd_missions: int

    @property
    def fraction(self) -> float:
        return self.covered_cells / self.total_cells if self.total_cells else 0.0


def odd_coverage(
    odd: OddSpec,
    results: list[MissionOddResult],
    min_missions_per_cell: int = 1,
    min_in_odd_fraction: float = 0.95,
) -> OddCoverage:
    """Coverage of the ODD cells ``environment x terrain x lighting band``.

    A cell counts as covered when at least ``min_missions_per_cell`` missions
    that stayed inside the ODD (see :func:`is_in_odd`) ran in it. Missions
    whose cell is not part of the ODD are counted in ``out_of_odd_missions``.
    """
    cells = list(product(odd.environment.allowed, odd.terrain.allowed, odd.lighting_bands()))
    counts = dict.fromkeys(cells, 0)
    outside = 0
    for r in results:
        if r.cell not in counts:
            outside += 1
        elif is_in_odd(r, min_in_odd_fraction):
            counts[r.cell] += 1
    uncovered = [c for c in cells if counts[c] < min_missions_per_cell]
    return OddCoverage(len(cells), len(cells) - len(uncovered), counts, uncovered, outside)


def is_in_odd(result: MissionOddResult, min_in_odd_fraction: float = 0.95) -> bool:
    """True if a mission spent at least ``min_in_odd_fraction`` of autonomous time in the ODD."""
    return result.in_odd_fraction >= min_in_odd_fraction
