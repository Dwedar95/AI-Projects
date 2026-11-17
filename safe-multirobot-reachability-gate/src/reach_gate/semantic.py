"""Semantic layer: a VLM/LLM proposes constraints and task allocation as structured JSON.

The language model is never trusted with safety. Its output is parsed against a strict
schema and merged into the hard constraints in a way that can only tighten them. The
reachability gate then checks every motion command against the merged constraints.
"""

from __future__ import annotations

import json
import os
from typing import Literal, Protocol

import numpy as np
from pydantic import BaseModel, Field, ValidationError
from scipy.optimize import linear_sum_assignment

from .gate import Box, SafetyConstraints


class RobotInfo(BaseModel):
    robot_id: str
    kind: Literal["aerial", "ground"]
    capabilities: list[str]
    position: tuple[float, float, float]
    busy: bool = False


class Task(BaseModel):
    task_id: str
    capability: str
    location: tuple[float, float]


class Observation(BaseModel):
    """Something the perception model reports, e.g. a VLM caption with a location."""

    label: str
    position: tuple[float, float]
    half_size: tuple[float, float] = (0.3, 0.3)
    confidence: float = Field(1.0, ge=0.0, le=1.0)


class SceneReport(BaseModel):
    robots: list[RobotInfo]
    tasks: list[Task]
    observations: list[Observation]


class KeepOutZone(BaseModel):
    label: str
    center: tuple[float, float]
    half_size: tuple[float, float] = Field(..., description="half extents in x and y, metres")
    height: float = Field(..., gt=0.0, le=50.0)
    reason: str = ""


class Directive(BaseModel):
    robot_id: str
    task_id: str | None = None
    goal: tuple[float, float] | None = None
    max_speed: float | None = Field(None, gt=0.0)
    altitude_band: tuple[float, float] | None = None


class SemanticPlan(BaseModel):
    keep_out: list[KeepOutZone] = []
    directives: list[Directive] = []
    min_separation: float | None = Field(None, gt=0.0, le=10.0)
    rationale: str = ""


class SemanticProvider(Protocol):
    """Returns the model's raw text answer for a scene; parsing happens outside."""

    def propose(self, scene: SceneReport) -> str: ...


def parse_plan(text: str) -> SemanticPlan | None:
    """Validate a model answer. Returns None for anything that is not a valid plan."""
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        return SemanticPlan.model_validate(json.loads(text[start:end]))
    except (ValueError, ValidationError):
        return None


# Hazard classes known to the mock: (xy margin in metres, zone height, speed factor nearby).
HAZARD_RULES: dict[str, tuple[float, float, float]] = {
    "person": (1.0, 4.0, 0.5),
    "spill": (0.3, 0.3, 1.0),
    "debris": (0.3, 0.5, 1.0),
    "open_pit": (0.5, 0.2, 0.7),
}
UNKNOWN_HAZARD = (0.6, 4.0, 0.7)


class MockSemanticProvider:
    """Deterministic offline stand-in for a VLM/LLM, used by tests and the demo.

    It applies fixed rules: every observed hazard becomes a keep-out zone with a
    class-dependent margin and height, robots near a person are slowed down, the
    aerial robot's altitude floor is raised when people are present, and open tasks
    are assigned to idle robots with the Hungarian algorithm on travel distance,
    respecting the required capability. The answer is returned as JSON text, the same
    way a language model would return it.
    """

    slow_radius: float = 4.0
    base_speed: dict[str, float]

    def __init__(self, base_speed: dict[str, float] | None = None,
                 min_confidence: float = 0.3) -> None:
        self.base_speed = base_speed or {}
        self.min_confidence = min_confidence

    def propose(self, scene: SceneReport) -> str:
        keep_out, notes = [], []
        hazards = [o for o in scene.observations if o.confidence >= self.min_confidence]
        for ob in hazards:
            margin, height, _ = HAZARD_RULES.get(ob.label, UNKNOWN_HAZARD)
            keep_out.append(KeepOutZone(
                label=ob.label, center=ob.position,
                half_size=(ob.half_size[0] + margin, ob.half_size[1] + margin),
                height=height, reason=f"{ob.label} observed with confidence {ob.confidence:.2f}"))
        people = [o for o in hazards if o.label == "person"]
        directives: dict[str, Directive] = {r.robot_id: Directive(robot_id=r.robot_id)
                                            for r in scene.robots}
        for robot in scene.robots:
            d = directives[robot.robot_id]
            factor = 1.0
            for ob in hazards:
                _, _, slow = HAZARD_RULES.get(ob.label, UNKNOWN_HAZARD)
                dist = float(np.hypot(robot.position[0] - ob.position[0],
                                      robot.position[1] - ob.position[1]))
                if dist <= self.slow_radius:
                    factor = min(factor, slow)
            if factor < 1.0 and robot.robot_id in self.base_speed:
                d.max_speed = round(self.base_speed[robot.robot_id] * factor, 3)
                notes.append(f"{robot.robot_id} slowed to {d.max_speed} m/s near a hazard")
            if robot.kind == "aerial" and people:
                d.altitude_band = (2.0, 3.0)
        # Task allocation: minimise total distance, capability mismatches are forbidden.
        idle = [r for r in scene.robots if not r.busy]
        if idle and scene.tasks:
            cost = np.full((len(idle), len(scene.tasks)), 1e6)
            for i, robot in enumerate(idle):
                for j, task in enumerate(scene.tasks):
                    if task.capability in robot.capabilities:
                        cost[i, j] = np.hypot(robot.position[0] - task.location[0],
                                              robot.position[1] - task.location[1])
            for i, j in zip(*linear_sum_assignment(cost), strict=True):
                if cost[i, j] < 1e6:
                    d = directives[idle[i].robot_id]
                    d.task_id, d.goal = scene.tasks[j].task_id, scene.tasks[j].location
                    notes.append(f"{scene.tasks[j].task_id} -> {idle[i].robot_id}")
        plan = SemanticPlan(keep_out=keep_out, directives=list(directives.values()),
                            min_separation=0.8 if people else None,
                            rationale="; ".join(notes) or "no hazards, no open tasks")
        return plan.model_dump_json()


SYSTEM_PROMPT = (
    "You coordinate a team of aerial and ground robots. Given a JSON scene report, answer "
    "with one JSON object and nothing else, matching this JSON schema:\n{schema}\n"
    "Add a keep-out zone for every hazard, assign each open task to one idle robot that has "
    "the required capability, and lower max_speed near people. A separate verified safety "
    "layer checks every motion command, so never claim that a motion is safe."
)


class OpenAIChatProvider:
    """OPTIONAL adapter for OpenAI or Azure OpenAI chat models. UNTESTED in this repository.

    It needs the ``openai`` package and network access, neither of which is available
    in the offline test environment, so this class has never been executed here. It
    only builds the request and returns the raw text; validation is done by
    :func:`parse_plan` exactly as for the mock.

    Environment variables: ``AZURE_OPENAI_ENDPOINT``, ``AZURE_OPENAI_API_KEY``,
    ``AZURE_OPENAI_DEPLOYMENT`` and optionally ``AZURE_OPENAI_API_VERSION`` for Azure,
    or ``OPENAI_API_KEY`` and optionally ``OPENAI_MODEL`` for OpenAI.
    """

    def __init__(self) -> None:
        try:
            import openai  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("OpenAIChatProvider needs: pip install openai") from exc
        if os.environ.get("AZURE_OPENAI_ENDPOINT"):  # pragma: no cover - needs network
            self._client = openai.AzureOpenAI(
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01"))
            self._model = os.environ["AZURE_OPENAI_DEPLOYMENT"]
        else:  # pragma: no cover - needs network
            self._client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            self._model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    def propose(self, scene: SceneReport) -> str:  # pragma: no cover - needs network
        schema = json.dumps(SemanticPlan.model_json_schema())
        response = self._client.chat.completions.create(
            model=self._model, temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SYSTEM_PROMPT.format(schema=schema)},
                      {"role": "user", "content": scene.model_dump_json()}])
        return response.choices[0].message.content or ""


def merge_constraints(base: SafetyConstraints, plan: SemanticPlan,
                      robot_speeds: dict[str, float]) -> SafetyConstraints:
    """Combine hard base constraints with a semantic plan. The result is never looser.

    Keep-out zones are added to the obstacle list, the separation is the larger of the
    two values, altitude bands are intersected and speed limits take the minimum. A plan
    cannot remove an obstacle, widen a corridor or reduce the separation.
    """
    obstacles = list(base.obstacles)
    for zone in plan.keep_out:
        obstacles.append(Box(
            (zone.center[0] - zone.half_size[0], zone.center[1] - zone.half_size[1], 0.0),
            (zone.center[0] + zone.half_size[0], zone.center[1] + zone.half_size[1], zone.height),
            f"keep-out:{zone.label}"))
    bands = dict(base.altitude_bands)
    limits = dict(base.speed_limits)
    for d in plan.directives:
        if d.altitude_band is not None and d.robot_id in bands:
            lo = max(bands[d.robot_id][0], min(d.altitude_band))
            hi = min(bands[d.robot_id][1], max(d.altitude_band))
            if lo < hi:  # an empty intersection would be an unusable request, keep the base
                bands[d.robot_id] = (lo, hi)
        if d.max_speed is not None:
            current = limits.get(d.robot_id, robot_speeds.get(d.robot_id, np.inf))
            limits[d.robot_id] = min(current, d.max_speed)
    separation = max(base.min_separation, plan.min_separation or 0.0)
    return base.tightened(obstacles=tuple(obstacles), altitude_bands=bands,
                          speed_limits=limits, min_separation=separation)
