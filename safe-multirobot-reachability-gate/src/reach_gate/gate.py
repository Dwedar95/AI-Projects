"""Safety gate: verify a candidate command by reachability, then publish or withhold.

A command is published only if the whole reachable tube of "apply the command for one
supervisory cycle, then stop" (a) misses every obstacle and keep-out zone, (b) stays in
the robot's corridor and altitude band, and (c) keeps the required 3D separation from
the tubes of all other agents. Otherwise it is withheld and never reaches the robot.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .bus import MessageBus
from .reachability import ReachTube, RobotModel, build_tube, contingency_inputs
from .zonotope import Zonotope


@dataclass(frozen=True)
class Box:
    """Axis-aligned 3D box."""

    lower: tuple[float, float, float]
    upper: tuple[float, float, float]
    label: str = ""

    def as_zonotope(self) -> Zonotope:
        return Zonotope.from_box(np.asarray(self.lower), np.asarray(self.upper))

    def intersects_box(self, lower: np.ndarray, upper: np.ndarray) -> bool:
        return bool(np.all(np.asarray(self.lower) <= upper)
                    and np.all(lower <= np.asarray(self.upper)))


@dataclass(frozen=True)
class SafetyConstraints:
    """Everything the gate checks. All fields are hard constraints."""

    workspace: Box
    obstacles: tuple[Box, ...] = ()
    corridors: dict[str, Box] = field(default_factory=dict)             # per robot
    altitude_bands: dict[str, tuple[float, float]] = field(default_factory=dict)
    speed_limits: dict[str, float] = field(default_factory=dict)
    min_separation: float = 0.5

    def containment_box(self, robot_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Region the robot's body must stay in: workspace, corridor and altitude band."""
        lo = np.asarray(self.workspace.lower, dtype=np.float64).copy()
        hi = np.asarray(self.workspace.upper, dtype=np.float64).copy()
        if robot_id in self.corridors:
            lo = np.maximum(lo, self.corridors[robot_id].lower)
            hi = np.minimum(hi, self.corridors[robot_id].upper)
        if robot_id in self.altitude_bands:
            lo[2] = max(lo[2], self.altitude_bands[robot_id][0])
            hi[2] = min(hi[2], self.altitude_bands[robot_id][1])
        return lo, hi

    def tightened(self, **changes) -> SafetyConstraints:
        return replace(self, **changes)


@dataclass(frozen=True)
class Violation:
    kind: str          # "obstacle", "containment", "separation" or "speed_limit"
    step: int          # index of the first violating tube segment, -1 for command checks
    detail: str


@dataclass
class Verdict:
    robot_id: str
    command: np.ndarray
    safe: bool
    violations: list[Violation]
    tube: ReachTube


@dataclass
class GateStats:
    verifications: int = 0
    published: int = 0
    withheld: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)

    def record(self, verdict: Verdict) -> None:
        self.verifications += 1
        if verdict.safe:
            self.published += 1
        else:
            self.withheld += 1
            for kind in {v.kind for v in verdict.violations}:
                self.by_kind[kind] = self.by_kind.get(kind, 0) + 1


class SafetyGate:
    """Verifies commands and is the only component that publishes on the command topic."""

    def __init__(self, bus: MessageBus | None = None, horizon: int = 15, commit_steps: int = 5,
                 stop_at_first_violation: bool = True) -> None:
        if not 0 < commit_steps <= horizon:
            raise ValueError("commit_steps must be in 1..horizon")
        self.bus = bus
        self.horizon = horizon
        self.commit_steps = commit_steps
        self.stop_at_first = stop_at_first_violation
        self.stats = GateStats()

    def verify(self, robot_id: str, model: RobotModel, x0: Zonotope, command: np.ndarray,
               constraints: SafetyConstraints, other_tubes: dict[str, ReachTube] | None = None,
               ) -> Verdict:
        """Pure check, no side effects: reachable tube against all constraints."""
        command = np.asarray(command, dtype=np.float64)
        tube = build_tube(robot_id, model, x0,
                          contingency_inputs(model, command, self.horizon, self.commit_steps))
        violations: list[Violation] = []
        limit = constraints.speed_limits.get(robot_id)
        speed = abs(command[0]) if model.kind == "ground" else float(np.linalg.norm(command))
        if limit is not None and speed > limit + 1e-9:
            violations.append(Violation("speed_limit", -1, f"{speed:.2f} > {limit:.2f} m/s"))
        lo_box, hi_box = constraints.containment_box(robot_id)
        obstacles = [(ob, ob.as_zonotope()) for ob in constraints.obstacles]
        sep = constraints.min_separation
        others = other_tubes or {}
        for k, occ in enumerate(tube.occupancy):
            if violations and self.stop_at_first:
                break
            lo, hi = occ.interval_hull()
            if np.any(lo < lo_box - 1e-9) or np.any(hi > hi_box + 1e-9):
                violations.append(Violation("containment", k, "leaves corridor or altitude band"))
                continue
            for ob, ob_zono in obstacles:
                if ob.intersects_box(lo, hi) and occ.intersects(ob_zono):
                    violations.append(Violation("obstacle", k, ob.label or "obstacle"))
                    break
            # Separation: grow this robot's set by the required distance in every axis.
            # If the grown set misses the other set, all point pairs are farther apart
            # than ``sep`` in the infinity norm and therefore in the Euclidean norm.
            grown = occ + Zonotope.from_box([-sep] * 3, [sep] * 3)
            for other_id, other in others.items():
                if other_id == robot_id or k >= other.steps:
                    continue
                if grown.intersects(other.occupancy[k]):
                    violations.append(Violation("separation", k, other_id))
                    break
        return Verdict(robot_id, command, not violations, violations, tube)

    def submit(self, robot_id: str, model: RobotModel, x0: Zonotope, command: np.ndarray,
               constraints: SafetyConstraints, other_tubes: dict[str, ReachTube] | None = None,
               ) -> Verdict:
        """Verify and act: publish the command if it is safe, withhold it otherwise."""
        verdict = self.verify(robot_id, model, x0, command, constraints, other_tubes)
        self.stats.record(verdict)
        if self.bus is not None:
            if verdict.safe:
                self.bus.publish(f"fleet/{robot_id}/cmd/verified",
                                 {"command": verdict.command.tolist(), "horizon": self.horizon})
            else:
                first = verdict.violations[0]
                self.bus.publish(f"fleet/{robot_id}/cmd/withheld",
                                 {"command": verdict.command.tolist(), "reason": first.kind,
                                  "step": first.step, "detail": first.detail})
        return verdict
