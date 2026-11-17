"""Multi-robot simulation: quadrotor, quadruped-like robot and ground vehicle.

A supervisor runs fixed-rate cycles. In each cycle the semantic layer may update the
constraints and task allocation, every robot's proposer suggests commands, the safety
gate verifies them in rank order and publishes the first safe one, and the plants are
integrated with bounded random disturbances. Independent ground-truth monitors record
what actually happened, so the summary does not rely on the gate's own bookkeeping.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .bus import Message, MessageBus
from .gate import Box, SafetyConstraints, SafetyGate
from .proposer import plan_route, propose_aerial, propose_ground, unsafe_candidates
from .reachability import (
    QuadrotorModel,
    ReachTube,
    RobotModel,
    UnicycleModel,
    build_tube,
    shift_tube,
)
from .semantic import (
    KeepOutZone,
    MockSemanticProvider,
    Observation,
    RobotInfo,
    SceneReport,
    SemanticProvider,
    Task,
    merge_constraints,
    parse_plan,
)


@dataclass
class RobotAgent:
    """Plant state plus the command interface of one robot."""

    robot_id: str
    model: RobotModel
    state: np.ndarray
    cruise_altitude: float = 0.0
    goal: np.ndarray | None = None
    task_id: str | None = None
    command: np.ndarray = field(default_factory=lambda: np.zeros(0))
    tube: ReachTube | None = None
    route: list[np.ndarray] = field(default_factory=list)
    route_key: tuple | None = None

    def __post_init__(self) -> None:
        self.command = self.model.stop_command()

    def attach(self, bus: MessageBus) -> None:
        """The robot only ever executes what arrives on its verified or fallback topic."""
        bus.subscribe(f"fleet/{self.robot_id}/cmd/verified", self._on_command)
        bus.subscribe(f"fleet/{self.robot_id}/cmd/fallback", self._on_command)

    def _on_command(self, msg: Message) -> None:
        self.command = np.asarray(msg.payload["command"], dtype=np.float64)

    def next_waypoint(self, position: np.ndarray, switch_radius: float = 0.7) -> np.ndarray:
        """Current route waypoint; waypoints are dropped once the robot is close."""
        while len(self.route) > 1 and \
                np.linalg.norm(self.route[0] - position[:2]) < switch_radius:
            self.route.pop(0)
        target = self.route[0] if self.route else np.asarray(self.goal, dtype=np.float64)
        if self.model.kind == "aerial":
            return np.array([target[0], target[1], self.cruise_altitude])
        return target

    def body_box(self) -> tuple[np.ndarray, np.ndarray]:
        p, h = self.model.position(self.state), self.model.body_half_extent()
        return p - h, p + h


@dataclass
class ScenarioConfig:
    seed: int = 0
    cycles: int = 120
    horizon: int = 15
    commit_steps: int = 5
    semantic_period: int = 4
    arrival_tolerance: float = 0.45
    inject_every: int = 4            # every n-th cycle one robot gets an unsafe primary command
    late_observation_cycle: int = 30
    route_margin: float = 0.25       # clearance added to the footprint for route hints


@dataclass
class RunSummary:
    cycles: int = 0
    sim_time_s: float = 0.0
    verifications: int = 0
    published: int = 0
    withheld: int = 0
    withheld_by_kind: dict[str, int] = field(default_factory=dict)
    safe_stops: int = 0
    contingency_stops: int = 0
    injected_unsafe: int = 0
    injected_withheld: int = 0
    tasks_total: int = 0
    tasks_done: int = 0
    semantic_updates: int = 0
    semantic_rejected: int = 0
    deferred_zones: int = 0
    min_body_gap_m: float = float("inf")
    min_center_distance_m: float = float("inf")
    required_separation_m: float = 0.0
    collisions: int = 0
    containment_violations: int = 0
    bus_messages: int = 0
    mean_verification_ms: float = 0.0

    def lines(self) -> list[str]:
        kinds = ", ".join(f"{k}: {v}" for k, v in sorted(self.withheld_by_kind.items())) or "none"
        return [
            f"supervisory cycles        : {self.cycles} ({self.sim_time_s:.1f} s simulated)",
            f"verifications             : {self.verifications}",
            f"commands published        : {self.published} (of which safe stops: "
            f"{self.safe_stops})",
            f"commands withheld         : {self.withheld} ({kinds})",
            f"contingency stops         : {self.contingency_stops}",
            f"injected unsafe commands  : {self.injected_unsafe}, withheld: "
            f"{self.injected_withheld}",
            f"tasks completed           : {self.tasks_done} of {self.tasks_total}",
            f"semantic plan updates     : {self.semantic_updates} "
            f"(rejected answers: {self.semantic_rejected}, deferred keep-out zones: "
            f"{self.deferred_zones})",
            f"minimum body gap          : {self.min_body_gap_m:.2f} m "
            f"(required {self.required_separation_m:.2f} m)",
            f"minimum centre distance   : {self.min_center_distance_m:.2f} m",
            f"ground-truth collisions   : {self.collisions}",
            f"containment violations    : {self.containment_violations}",
            f"bus messages              : {self.bus_messages}",
            f"mean verification time    : {self.mean_verification_ms:.2f} ms",
        ]


def default_world() -> tuple[SafetyConstraints, list[RobotAgent], list[Task], list[Observation]]:
    """A 20 m x 12 m yard with a tall rack, a low crate and three robots."""
    base = SafetyConstraints(
        workspace=Box((0.0, 0.0, 0.0), (20.0, 12.0, 4.0), "workspace"),
        obstacles=(
            Box((6.0, 4.0, 0.0), (8.0, 7.5, 3.6), "rack"),
            Box((12.0, 6.5, 0.0), (13.5, 8.0, 0.8), "crate"),
        ),
        altitude_bands={"uav": (1.5, 3.2)},
        min_separation=0.5,
    )
    robots = [
        RobotAgent("uav", QuadrotorModel(), np.array([1.5, 6.0, 2.5, 0.0, 0.0, 0.0]),
                   cruise_altitude=2.5),
        RobotAgent("ugv", UnicycleModel(max_speed=1.2, capabilities=("ground", "carry")),
                   np.array([1.5, 3.0, 0.0])),
        RobotAgent("dog", UnicycleModel(max_speed=0.8, max_yaw_rate=1.5, footprint=0.3,
                                        height=0.5, input_error=(0.05, 0.08),
                                        disturbance=(0.008, 0.008, 0.01),
                                        capabilities=("ground", "legged")),
                   np.array([1.5, 9.0, 0.0])),
    ]
    tasks = [
        Task(task_id="inspect-roof", capability="aerial", location=(17.0, 9.5)),
        Task(task_id="deliver-parts", capability="carry", location=(17.5, 5.5)),
        Task(task_id="patrol-north", capability="legged", location=(15.0, 10.5)),
        Task(task_id="inspect-gate", capability="aerial", location=(3.0, 2.0)),
        Task(task_id="return-pallet", capability="carry", location=(3.0, 9.5)),
        Task(task_id="patrol-south", capability="legged", location=(10.0, 2.0)),
    ]
    observations = [
        Observation(label="spill", position=(10.5, 9.8), half_size=(0.5, 0.4), confidence=0.9),
        Observation(label="person", position=(15.5, 2.5), half_size=(0.3, 0.3), confidence=0.95),
    ]
    return base, robots, tasks, observations


class Simulation:
    """Supervisor, plants and monitors wired together over the message bus."""

    def __init__(self, config: ScenarioConfig | None = None,
                 provider: SemanticProvider | None = None,
                 world: tuple | None = None) -> None:
        self.cfg = config or ScenarioConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.base, self.robots, self.open_tasks, self.observations = world or default_world()
        self.bus = MessageBus()
        self.gate = SafetyGate(self.bus, self.cfg.horizon, self.cfg.commit_steps)
        self.provider = provider or MockSemanticProvider(
            {r.robot_id: r.model.max_speed for r in self.robots})
        self.constraints = self.base
        self.summary = RunSummary(tasks_total=len(self.open_tasks))
        self._pending: dict[str, Task] = {}
        self._verify_time = 0.0
        for robot in self.robots:
            robot.attach(self.bus)

    # ------------------------------------------------------------------ semantic layer
    def _scene(self) -> SceneReport:
        return SceneReport(
            robots=[RobotInfo(robot_id=r.robot_id, kind=r.model.kind,
                              capabilities=list(r.model.capabilities),
                              position=tuple(float(v) for v in r.model.position(r.state)),
                              busy=r.goal is not None) for r in self.robots],
            tasks=list(self.open_tasks), observations=list(self.observations))

    def _semantic_update(self) -> None:
        scene = self._scene()
        self.bus.publish("fleet/semantic/scene", scene.model_dump())
        plan = parse_plan(self.provider.propose(scene))
        if plan is None:
            self.summary.semantic_rejected += 1  # keep the previous constraints
            return
        self.summary.semantic_updates += 1
        speeds = {r.robot_id: r.model.max_speed for r in self.robots}
        usable = [z for z in plan.keep_out if not self._zone_conflicts(z)]
        self.summary.deferred_zones += len(plan.keep_out) - len(usable)
        plan = plan.model_copy(update={"keep_out": usable})
        self.constraints = merge_constraints(self.base, plan, speeds)
        self.bus.publish("fleet/semantic/plan", plan.model_dump(), retain=True)
        by_id = {r.robot_id: r for r in self.robots}
        open_by_id = {t.task_id: t for t in self.open_tasks}
        for d in plan.directives:
            robot = by_id.get(d.robot_id)
            task = open_by_id.get(d.task_id or "")
            # Allocation is advisory: it is accepted only for an idle robot with the
            # required capability and for a task that is really open.
            if robot is None or task is None or robot.goal is not None:
                continue
            if task.capability not in robot.model.capabilities:
                continue
            robot.goal, robot.task_id = np.asarray(task.location, dtype=np.float64), task.task_id
            self.open_tasks.remove(task)
            del open_by_id[task.task_id]
            self.bus.publish(f"fleet/{robot.robot_id}/task/assigned",
                             {"task_id": task.task_id, "goal": list(task.location)})

    def _zone_conflicts(self, zone: KeepOutZone) -> bool:
        """True if a new keep-out zone overlaps what a robot is already committed to.

        A zone declared on top of a robot, or across the stop manoeuvre it has already
        been cleared for, cannot be honoured without breaking that commitment. Such a
        zone is deferred and tried again at the next semantic update. Zones that are
        already active stay active.
        """
        label = f"keep-out:{zone.label}"
        box = Box((zone.center[0] - zone.half_size[0], zone.center[1] - zone.half_size[1], 0.0),
                  (zone.center[0] + zone.half_size[0], zone.center[1] + zone.half_size[1],
                   zone.height), label)
        if box in self.constraints.obstacles:
            return False
        for robot in self.robots:
            lo, hi = robot.body_box()
            if box.intersects_box(lo - 0.1, hi + 0.1):
                return True
            if robot.tube is not None:
                for occ in robot.tube.occupancy[self.cfg.commit_steps:]:
                    if box.intersects_box(*occ.interval_hull()):
                        return True
        return False

    # ------------------------------------------------------------------ supervision
    def _update_route(self, robot: RobotAgent, measured: np.ndarray) -> None:
        """Replan the waypoint hint when the goal or the constraints have changed."""
        key = (robot.task_id, id(self.constraints))
        if robot.route_key == key:
            return
        robot.route_key = key
        lo_box, hi_box = self.constraints.containment_box(robot.robot_id)
        aerial = robot.model.kind == "aerial"
        # A box only blocks the aerial robot if it reaches into its altitude band.
        floor = lo_box[2] - self.constraints.min_separation if aerial else -np.inf
        blocked = [(np.asarray(ob.lower[:2]), np.asarray(ob.upper[:2]))
                   for ob in self.constraints.obstacles if ob.upper[2] > floor]
        inflate = robot.model.footprint + self.cfg.route_margin
        robot.route = plan_route(measured[:2], robot.goal, blocked,
                                 (lo_box[:2], hi_box[:2]), inflate)

    def _proposals(self, robot: RobotAgent, measured: np.ndarray) -> list[np.ndarray]:
        cap = self.constraints.speed_limits.get(robot.robot_id, robot.model.max_speed)
        if robot.goal is None:
            if robot.model.kind == "aerial":  # hold position while idle
                hold = np.array([measured[0], measured[1], robot.cruise_altitude])
                return propose_aerial(robot.model, measured, hold, cap)[:1]
            return []
        self._update_route(robot, measured)
        target = robot.next_waypoint(measured)
        if robot.model.kind == "aerial":
            return propose_aerial(robot.model, measured, target, cap)
        return propose_ground(robot.model, measured, target, cap)

    def nominal_violation(self, robot: RobotAgent, command: np.ndarray) -> bool:
        """Ground-truth style check: does the disturbance-free rollout break a static rule?"""
        lo_box, hi_box = self.constraints.containment_box(robot.robot_id)
        h = robot.model.body_half_extent()
        x = robot.state.copy()
        for k in range(self.cfg.horizon):
            u = command if k < self.cfg.commit_steps else robot.model.stop_command()
            x = robot.model.step(x, u)
            p = robot.model.position(x)
            if np.any(p - h < lo_box) or np.any(p + h > hi_box):
                return True
            if any(ob.intersects_box(p - h, p + h) for ob in self.constraints.obstacles):
                return True
        return False

    def _supervise(self, cycle: int) -> None:
        committed: dict[str, ReachTube] = {}
        # What every robot is still cleared to do: the rest of its last verified tube,
        # re-indexed to the current time. Robots handled later in this cycle are
        # represented by this until they commit to something new.
        shift = self.cfg.commit_steps if cycle > 0 else 0
        remaining = {r.robot_id: shift_tube(r.tube, r.model, shift) for r in self.robots}
        inject_robot = None
        if self.cfg.inject_every and cycle % self.cfg.inject_every == self.cfg.inject_every - 1:
            inject_robot = self.robots[(cycle // self.cfg.inject_every) % len(self.robots)]
        for robot in self.robots:
            model = robot.model
            noise = self.rng.uniform(-1.0, 1.0, model.state_dim) * np.asarray(model.state_noise)
            measured = robot.state + noise
            x0 = model.measurement_set(measured)
            self.bus.publish(f"fleet/{robot.robot_id}/state", {"state": measured.tolist()})
            others = {r.robot_id: committed.get(r.robot_id, remaining[r.robot_id])
                      for r in self.robots if r is not robot}
            candidates = self._proposals(robot, measured)
            injected = None
            if robot is inject_robot:
                injected = next((u for u in unsafe_candidates(model)
                                 if self.nominal_violation(robot, u)), None)
                if injected is not None:
                    candidates = [injected, *candidates]
                    self.summary.injected_unsafe += 1
            candidates.append(model.stop_command())
            chosen = None
            for i, command in enumerate(candidates):
                self.bus.publish(f"fleet/{robot.robot_id}/cmd/proposed",
                                 {"command": np.asarray(command).tolist(), "rank": i})
                t0 = time.perf_counter()
                verdict = self.gate.submit(robot.robot_id, model, x0, command,
                                           self.constraints, others)
                self._verify_time += time.perf_counter() - t0
                if injected is not None and i == 0 and not verdict.safe:
                    self.summary.injected_withheld += 1
                if verdict.safe:
                    chosen = verdict
                    break
            if chosen is not None:
                robot.tube = chosen.tube
                if i == len(candidates) - 1:
                    self.summary.safe_stops += 1
            else:
                # Nothing verified, not even a fresh stop: continue the stop tail of the
                # tube that was verified in an earlier cycle.
                self.summary.contingency_stops += 1
                self.bus.publish(f"fleet/{robot.robot_id}/cmd/fallback",
                                 {"command": model.stop_command().tolist()})
                robot.tube = remaining[robot.robot_id]
            committed[robot.robot_id] = robot.tube

    # ------------------------------------------------------------------ plant and monitors
    def _monitor(self) -> None:
        s = self.summary
        boxes = {r.robot_id: r.body_box() for r in self.robots}
        for robot in self.robots:
            lo, hi = boxes[robot.robot_id]
            lo_box, hi_box = self.constraints.containment_box(robot.robot_id)
            if np.any(lo < lo_box) or np.any(hi > hi_box):
                s.containment_violations += 1
            if any(ob.intersects_box(lo, hi) for ob in self.constraints.obstacles):
                s.collisions += 1
        ids = list(boxes)
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                (lo_a, hi_a), (lo_b, hi_b) = boxes[ids[a]], boxes[ids[b]]
                gap = float(np.max(np.maximum(lo_a - hi_b, lo_b - hi_a)))
                centre = float(np.linalg.norm(0.5 * (lo_a + hi_a) - 0.5 * (lo_b + hi_b)))
                s.min_body_gap_m = min(s.min_body_gap_m, gap)
                s.min_center_distance_m = min(s.min_center_distance_m, centre)

    def _advance(self) -> None:
        for _ in range(self.cfg.commit_steps):
            for robot in self.robots:
                robot.state = robot.model.simulate_step(robot.state, robot.command, self.rng)
            self._monitor()
        for robot in self.robots:
            if robot.goal is None:
                continue
            p = robot.model.position(robot.state)
            if np.linalg.norm(p[:2] - robot.goal) <= self.cfg.arrival_tolerance:
                self.summary.tasks_done += 1
                self.bus.publish(f"fleet/{robot.robot_id}/task/done", {"task_id": robot.task_id})
                robot.goal, robot.task_id, robot.route_key = None, None, None

    def run(self) -> RunSummary:
        """Run the configured number of cycles and return the measured summary."""
        cfg = self.cfg
        for robot in self.robots:  # every robot starts parked with a verified stop tube
            x0 = robot.model.measurement_set(robot.state)
            robot.tube = build_tube(robot.robot_id, robot.model, x0,
                                    [robot.model.stop_command()] * cfg.horizon)
        for cycle in range(cfg.cycles):
            if cycle == cfg.late_observation_cycle:
                self.observations.append(Observation(label="debris", position=(4.5, 9.6),
                                                     half_size=(0.3, 0.3), confidence=0.8))
            idle_with_work = any(
                r.goal is None and any(t.capability in r.model.capabilities
                                       for t in self.open_tasks) for r in self.robots)
            if cycle % cfg.semantic_period == 0 or idle_with_work \
                    or cycle == cfg.late_observation_cycle:
                self._semantic_update()
            self._supervise(cycle)
            self._advance()
            self.summary.cycles += 1
        s = self.summary
        s.sim_time_s = cfg.cycles * cfg.commit_steps * self.robots[0].model.dt
        stats = self.gate.stats
        s.verifications, s.published, s.withheld = stats.verifications, stats.published, \
            stats.withheld
        s.withheld_by_kind = dict(stats.by_kind)
        s.required_separation_m = self.constraints.min_separation
        s.bus_messages = len(self.bus.log)
        s.mean_verification_ms = 1e3 * self._verify_time / max(stats.verifications, 1)
        return s
