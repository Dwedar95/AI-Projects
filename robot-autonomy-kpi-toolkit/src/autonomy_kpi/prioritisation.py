"""Backlog prioritisation with RICE and WSJF, linked to KPIs.

Each backlog item names the KPIs it is expected to move. That link is what
connects the roadmap to the measurements: when a release gate fails on a KPI,
the items that target that KPI can be surfaced (and optionally boosted).

Scores:

* RICE = reach * impact * confidence / effort
* WSJF = (business_value + time_criticality + risk_reduction) / job_size

The inputs are human estimates. The scores order a discussion, they do not
replace it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .kpis import KPI_NAMES

Method = Literal["rice", "wsjf"]


class BacklogItem(BaseModel):
    """One feature or improvement candidate."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    kpis: list[str] = Field(min_length=1, description="KPIs this item is expected to move")
    # RICE inputs
    reach: float = Field(gt=0, description="Missions per quarter affected")
    impact: float = Field(
        gt=0, le=3, description="0.25 minimal, 0.5 low, 1 medium, 2 high, 3 massive"
    )
    confidence: float = Field(gt=0, le=1)
    effort: float = Field(gt=0, description="Person-months")
    # WSJF inputs (relative points, 1 to 20)
    business_value: float = Field(ge=1, le=20)
    time_criticality: float = Field(ge=1, le=20)
    risk_reduction: float = Field(ge=1, le=20)
    job_size: float = Field(ge=1, le=20)

    @model_validator(mode="after")
    def _known_kpis(self) -> BacklogItem:
        unknown = sorted(set(self.kpis) - set(KPI_NAMES))
        if unknown:
            raise ValueError(f"item {self.id}: unknown KPIs {unknown}")
        return self

    def rice(self) -> float:
        """RICE score."""
        return self.reach * self.impact * self.confidence / self.effort

    def wsjf(self) -> float:
        """Weighted Shortest Job First score (cost of delay divided by job size)."""
        return (self.business_value + self.time_criticality + self.risk_reduction) / self.job_size


def load_backlog(path: str | Path) -> list[BacklogItem]:
    """Load a backlog YAML file with a top-level ``backlog`` list. Ids must be unique."""
    with Path(path).open(encoding="utf-8") as fh:
        document = yaml.safe_load(fh)
    if not isinstance(document, dict) or not isinstance(document.get("backlog"), list):
        raise ValueError("backlog file must contain a top-level 'backlog' list")
    items = [BacklogItem.model_validate(entry) for entry in document["backlog"]]
    ids = [item.id for item in items]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"duplicate backlog ids: {duplicates}")
    return items


@dataclass(frozen=True)
class RankedItem:
    """A backlog item with its score and rank."""

    rank: int
    item: BacklogItem
    score: float
    base_score: float
    addresses: tuple[str, ...]


def rank_backlog(
    items: list[BacklogItem],
    method: Method = "rice",
    failing_kpis: tuple[str, ...] | list[str] = (),
    gate_boost: float = 1.0,
) -> list[RankedItem]:
    """Rank items by RICE or WSJF, highest score first.

    ``failing_kpis`` are KPIs whose release gate failed or warned. Items that
    target at least one of them are marked in ``addresses`` and their score is
    multiplied by ``gate_boost`` (1.0 means no boost). Ties keep backlog order.
    """
    if method not in ("rice", "wsjf"):
        raise ValueError(f"unknown method {method!r}")
    if gate_boost < 1.0:
        raise ValueError("gate_boost must be at least 1.0")
    failing = set(failing_kpis)
    scored = []
    for position, item in enumerate(items):
        base = item.rice() if method == "rice" else item.wsjf()
        addresses = tuple(k for k in item.kpis if k in failing)
        score = base * gate_boost if addresses else base
        scored.append((-score, position, item, base, addresses))
    scored.sort(key=lambda row: (row[0], row[1]))
    return [
        RankedItem(rank, item, -neg_score, base, addresses)
        for rank, (neg_score, _, item, base, addresses) in enumerate(scored, start=1)
    ]


def kpi_coverage(items: list[BacklogItem]) -> dict[str, list[str]]:
    """For every KPI, the ids of the backlog items that target it (empty list if none)."""
    coverage: dict[str, list[str]] = {name: [] for name in KPI_NAMES}
    for item in items:
        for kpi in item.kpis:
            coverage[kpi].append(item.id)
    return coverage
