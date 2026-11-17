"""Discrete-time reachability for the robot models used by the safety gate.

Each model propagates a zonotope of states through its dynamics for a sequence of
inputs, adding bounded actuation error and disturbance sets at every step. The result
over-approximates every state the simulated robot can reach under those bounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .zonotope import Zonotope, convex_hull_enclosure


@dataclass
class ReachTube:
    """Reachable sets of one robot over a horizon, on a common time grid."""

    robot_id: str
    dt: float
    inputs: list[np.ndarray]
    states: list[Zonotope]        # state sets, index 0 is the initial set
    occupancy: list[Zonotope]     # 3D space swept between step k and k + 1

    @property
    def steps(self) -> int:
        return len(self.occupancy)


@dataclass
class UnicycleModel:
    """Ground robot with state (x, y, heading) and input (speed, yaw rate).

    x+ = x + dt * (v cos(th), v sin(th), w) + disturbance, with the applied input
    equal to the command plus a bounded actuation error. Reachability linearises the
    dynamics around the nominal trajectory and adds a Lagrange remainder box that
    bounds the linearisation error, so the result is a guaranteed over-approximation.

    A command of exactly zero means "brake and hold": the robot stays where it is with
    no actuation error and no drift, in the plant and in the reachable set alike. This
    makes a stopped ground robot a true invariant, which the stop contingency relies on.
    """

    dt: float = 0.1
    max_speed: float = 1.0
    max_yaw_rate: float = 1.2
    input_error: tuple[float, float] = (0.03, 0.04)            # half-widths on (v, w)
    disturbance: tuple[float, float, float] = (0.004, 0.004, 0.004)
    state_noise: tuple[float, float, float] = (0.03, 0.03, 0.02)  # measurement bound
    footprint: float = 0.35      # half-width of the square body footprint, metres
    height: float = 0.6
    max_order: float = 60.0      # no reduction within the default horizon
    kind: str = "ground"
    capabilities: tuple[str, ...] = field(default_factory=lambda: ("ground",))

    state_dim = 3

    def clip(self, u: np.ndarray) -> np.ndarray:
        return np.array([np.clip(u[0], -self.max_speed, self.max_speed),
                         np.clip(u[1], -self.max_yaw_rate, self.max_yaw_rate)])

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Nominal one-step dynamics."""
        return np.array([x[0] + self.dt * u[0] * np.cos(x[2]),
                         x[1] + self.dt * u[0] * np.sin(x[2]),
                         x[2] + self.dt * u[1]])

    def jacobians(self, x: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """A = df/dx and B = df/du at (x, u)."""
        c, s = np.cos(x[2]), np.sin(x[2])
        a = np.array([[1.0, 0.0, -self.dt * u[0] * s], [0.0, 1.0, self.dt * u[0] * c],
                      [0.0, 0.0, 1.0]])
        b = np.array([[self.dt * c, 0.0], [self.dt * s, 0.0], [0.0, self.dt]])
        return a, b

    def lagrange_remainder(self, speed: float, d_heading: float, d_speed: float) -> float:
        """Bound on the second-order Taylor remainder of the x and y updates.

        With z = (heading, speed) the Hessian entries are bounded by dt * |v| and dt,
        so |r| <= 0.5 * dt * (|v|_max * d_heading^2 + 2 * d_heading * d_speed), where
        the d_ terms are the largest deviations from the linearisation point.
        """
        return 0.5 * self.dt * ((abs(speed) + d_speed) * d_heading ** 2
                                + 2.0 * d_heading * d_speed)

    def reach(self, x0: Zonotope, inputs: list[np.ndarray]) -> list[Zonotope]:
        """State sets after 0..len(inputs) steps for a set of initial states."""
        u_err = Zonotope.from_box(-np.asarray(self.input_error), np.asarray(self.input_error))
        w = Zonotope.from_box(-np.asarray(self.disturbance), np.asarray(self.disturbance))
        sets = [x0]
        current = x0
        for u in inputs:
            if not np.any(u):  # holding: the set does not change
                sets.append(current)
                continue
            nominal = current.center
            a, b = self.jacobians(nominal, u)
            d_heading = current.radius()[2]
            rem = self.lagrange_remainder(u[0], d_heading, self.input_error[0])
            deviation = Zonotope(np.zeros(3), current.generators)
            nxt = (deviation.linear_map(a) + u_err.linear_map(b) + w
                   + Zonotope.from_box([-rem, -rem, 0.0], [rem, rem, 0.0]))
            current = nxt.translate(self.step(nominal, u)).reduce_order(self.max_order)
            sets.append(current)
        return sets

    def simulate_step(self, x: np.ndarray, u: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """True plant step with sampled actuation error and disturbance inside the bounds."""
        if not np.any(u):
            return np.asarray(x, dtype=np.float64).copy()
        du = rng.uniform(-1.0, 1.0, 2) * np.asarray(self.input_error)
        w = rng.uniform(-1.0, 1.0, 3) * np.asarray(self.disturbance)
        return self.step(x, np.asarray(u) + du) + w

    def stop_command(self) -> np.ndarray:
        return np.zeros(2)

    def position(self, x: np.ndarray) -> np.ndarray:
        """Body centre in 3D."""
        return np.array([x[0], x[1], 0.5 * self.height])

    def body_half_extent(self) -> np.ndarray:
        return np.array([self.footprint, self.footprint, 0.5 * self.height])

    def occupancy(self, state_set: Zonotope) -> Zonotope:
        """3D set occupied by the body for every state in ``state_set``."""
        body = Zonotope.from_box([-self.footprint, -self.footprint, 0.0],
                                 [self.footprint, self.footprint, self.height])
        lift = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        # Body generators first: consecutive sets then pair up column by column, which
        # keeps the convex hull enclosure between them tight.
        return body + state_set.linear_map(lift)

    def measurement_set(self, measured: np.ndarray) -> Zonotope:
        """All true states consistent with a measurement under the noise bound."""
        n = np.asarray(self.state_noise)
        return Zonotope.from_box(measured - n, measured + n)


@dataclass
class QuadrotorModel:
    """Point-mass quadrotor with state (position, velocity) and a velocity command.

    The inner loop is modelled as a = gain * (v_cmd - v) + d with a bounded
    acceleration disturbance d (wind, tracking error). The closed loop is linear, so
    zonotope propagation is exact up to order reduction. Saturation is not modelled:
    commands are clipped to ``max_speed`` before they are verified.
    """

    dt: float = 0.1
    gain: float = 3.0
    max_speed: float = 1.5
    disturbance: tuple[float, float, float] = (0.25, 0.25, 0.15)   # m/s^2 half-widths
    state_noise: tuple[float, ...] = (0.03, 0.03, 0.03, 0.03, 0.03, 0.03)
    footprint: float = 0.3
    half_height: float = 0.15
    max_order: float = 60.0
    kind: str = "aerial"
    capabilities: tuple[str, ...] = field(default_factory=lambda: ("aerial",))

    state_dim = 6

    def __post_init__(self) -> None:
        dt, k, eye = self.dt, self.gain, np.eye(3)
        zero = np.zeros((3, 3))
        self.A = np.block([[eye, (dt - 0.5 * dt * dt * k) * eye], [zero, (1.0 - dt * k) * eye]])
        self.B = np.vstack([0.5 * dt * dt * k * eye, dt * k * eye])
        self.E = np.vstack([0.5 * dt * dt * eye, dt * eye])

    def clip(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64)
        n = np.linalg.norm(u)
        return u if n <= self.max_speed else u * (self.max_speed / n)

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.A @ x + self.B @ np.asarray(u)

    def reach(self, x0: Zonotope, inputs: list[np.ndarray]) -> list[Zonotope]:
        d = Zonotope.from_box(-np.asarray(self.disturbance), np.asarray(self.disturbance))
        d_mapped = d.linear_map(self.E)
        sets = [x0]
        current = x0
        for u in inputs:
            current = (current.linear_map(self.A).translate(self.B @ np.asarray(u))
                       + d_mapped).reduce_order(self.max_order)
            sets.append(current)
        return sets

    def simulate_step(self, x: np.ndarray, u: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        d = rng.uniform(-1.0, 1.0, 3) * np.asarray(self.disturbance)
        return self.step(x, u) + self.E @ d

    def stop_command(self) -> np.ndarray:
        return np.zeros(3)

    def position(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x[:3], dtype=np.float64)

    def body_half_extent(self) -> np.ndarray:
        return np.array([self.footprint, self.footprint, self.half_height])

    def occupancy(self, state_set: Zonotope) -> Zonotope:
        h = self.body_half_extent()
        return Zonotope.from_box(-h, h) + state_set.project([0, 1, 2])

    def measurement_set(self, measured: np.ndarray) -> Zonotope:
        n = np.asarray(self.state_noise)
        return Zonotope.from_box(measured - n, measured + n)


RobotModel = UnicycleModel | QuadrotorModel


def build_tube(robot_id: str, model: RobotModel, x0: Zonotope, inputs: list[np.ndarray],
               occupancy_order: float = 6.0) -> ReachTube:
    """Reachable tube with swept 3D occupancy between consecutive steps."""
    states = model.reach(x0, inputs)
    occ = [model.occupancy(s) for s in states]
    swept = [convex_hull_enclosure(a, b).reduce_order(occupancy_order)
             for a, b in zip(occ[:-1], occ[1:], strict=True)]
    return ReachTube(robot_id, model.dt, [np.asarray(u, dtype=np.float64) for u in inputs],
                     states, swept)


def contingency_inputs(model: RobotModel, command: np.ndarray, horizon: int,
                       commit_steps: int) -> list[np.ndarray]:
    """Input sequence that is verified: the command, then a stop until the horizon.

    Verifying the stop tail means a robot whose next command is withheld can always
    fall back to a manoeuvre that has already been verified.
    """
    stop = model.stop_command()
    command = np.asarray(command, dtype=np.float64)
    return [command] * commit_steps + [stop] * (horizon - commit_steps)


def shift_tube(tube: ReachTube, model: RobotModel, steps: int) -> ReachTube:
    """The same tube seen ``steps`` later, extended with stop inputs to keep its length."""
    if steps <= 0:
        return tube
    steps = min(steps, tube.steps)
    extra = [model.stop_command()] * steps
    tail_states = model.reach(tube.states[-1], extra)
    occ = [model.occupancy(s) for s in tail_states]
    swept = [convex_hull_enclosure(a, b).reduce_order(6.0)
             for a, b in zip(occ[:-1], occ[1:], strict=True)]
    return ReachTube(tube.robot_id, tube.dt, tube.inputs[steps:] + extra,
                     tube.states[steps:] + tail_states[1:], tube.occupancy[steps:] + swept)
