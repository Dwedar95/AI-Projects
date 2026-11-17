"""Mission log format: one JSON object per line (JSONL).

Every event has ``mission_id``, ``build``, ``t`` (seconds since mission start)
and ``type``. The remaining fields depend on the type, see
:data:`REQUIRED_FIELDS`.

Command events model a safety gate that sits between the planner and the
actuators: the planner emits ``command_proposed``, the gate answers with
``command_published`` (forwarded to the robot) or ``command_withheld``
(blocked, with a reason).
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

EventType = Literal[
    "mission_start",
    "command_proposed",
    "command_published",
    "command_withheld",
    "state",
    "intervention",
    "fault",
    "mission_end",
]

REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "mission_start": ("platform", "environment", "terrain", "planned_distance_m", "synthetic"),
    "command_proposed": ("command_id", "speed_mps"),
    "command_published": ("command_id",),
    "command_withheld": ("command_id", "reason"),
    "state": (
        "mode",
        "distance_m",
        "speed_mps",
        "localisation_error_m",
        "separation_m",
        "battery_soc",
        "lighting_lux",
        "link_latency_ms",
        "slope_deg",
    ),
    "intervention": ("kind", "duration_s", "cause"),
    "fault": ("code", "severity"),
    "mission_end": ("outcome", "distance_m", "duration_s"),
}

MODES = ("autonomous", "manual", "stopped")
OUTCOMES = ("success", "aborted", "failed")
INTERVENTION_KINDS = ("teleop_takeover", "remote_assist", "e_stop")
FAULT_SEVERITIES = ("minor", "major", "critical")

_ENUMS: dict[tuple[str, str], tuple[str, ...]] = {
    ("state", "mode"): MODES,
    ("mission_end", "outcome"): OUTCOMES,
    ("intervention", "kind"): INTERVENTION_KINDS,
    ("fault", "severity"): FAULT_SEVERITIES,
}


class Event(BaseModel):
    """One log event. Type-specific fields are kept as extra attributes."""

    model_config = ConfigDict(extra="allow")

    mission_id: str = Field(min_length=1)
    build: str = Field(min_length=1)
    t: float = Field(ge=0)
    type: EventType

    @model_validator(mode="after")
    def _check_fields(self) -> Event:
        extra = self.model_extra or {}
        missing = [f for f in REQUIRED_FIELDS[self.type] if f not in extra]
        if missing:
            raise ValueError(f"{self.type} event is missing fields: {missing}")
        for (etype, name), allowed in _ENUMS.items():
            if self.type == etype and extra[name] not in allowed:
                raise ValueError(f"{name}={extra[name]!r} is not one of {allowed}")
        return self

    def get(self, name: str, default: Any = None) -> Any:
        """Value of a type-specific field."""
        return (self.model_extra or {}).get(name, default)


def write_log(events: Iterable[Event], path: str | Path) -> int:
    """Write events as JSONL. Returns the number of lines written."""
    n = 0
    with Path(path).open("w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event.model_dump(), separators=(",", ":")) + "\n")
            n += 1
    return n


def iter_log(path: str | Path) -> Iterator[Event]:
    """Yield validated events from a JSONL file. Blank lines are skipped."""
    with Path(path).open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                yield Event.model_validate(json.loads(line))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc


def read_log(path: str | Path) -> list[Event]:
    """Read and validate a whole JSONL log."""
    return list(iter_log(path))


def group_by_mission(events: Iterable[Event]) -> dict[str, list[Event]]:
    """Group events by mission id, each group sorted by time (stable)."""
    groups: dict[str, list[Event]] = defaultdict(list)
    for event in events:
        groups[event.mission_id].append(event)
    return {mid: sorted(evts, key=lambda e: e.t) for mid, evts in groups.items()}


def validate_mission(events: list[Event]) -> list[str]:
    """Structural checks for the events of one mission. Returns a list of problems.

    Checks: single build, exactly one start (first) and one end (last),
    every proposed command resolved exactly once by the gate, no gate decision
    without a proposal, cumulative distance never decreasing.
    """
    problems: list[str] = []
    if not events:
        return ["mission has no events"]
    if len({e.build for e in events}) > 1:
        problems.append("events from more than one build")
    types = [e.type for e in events]
    if types.count("mission_start") != 1 or types[0] != "mission_start":
        problems.append("mission must begin with exactly one mission_start")
    if types.count("mission_end") != 1 or types[-1] != "mission_end":
        problems.append("mission must finish with exactly one mission_end")
    proposed: set[str] = set()
    resolved: set[str] = set()
    for e in events:
        cid = e.get("command_id")
        if e.type == "command_proposed":
            if cid in proposed:
                problems.append(f"command {cid} proposed twice")
            proposed.add(cid)
        elif e.type in ("command_published", "command_withheld"):
            if cid not in proposed:
                problems.append(f"gate decision for unknown command {cid}")
            if cid in resolved:
                problems.append(f"command {cid} resolved twice")
            resolved.add(cid)
    unresolved = proposed - resolved
    if unresolved:
        problems.append(f"{len(unresolved)} proposed commands have no gate decision")
    distances = [e.get("distance_m") for e in events if e.type == "state"]
    if any(b < a - 1e-9 for a, b in zip(distances, distances[1:], strict=False)):
        problems.append("cumulative distance decreases")
    return problems
