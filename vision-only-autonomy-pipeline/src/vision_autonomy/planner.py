"""DWA-style local planner: arc rollouts over a fixed set of curvature candidates."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .geometry import BevSpec


@dataclass
class PlannerConfig:
    num_candidates: int = 9
    max_curvature: float = 0.35          # 1/m
    horizon: float = 9.0                 # rollout arc length in metres
    step: float = 0.25                   # arc sampling step in metres
    vehicle_width: float = 1.0
    block_threshold: float = 0.35        # traversability below this blocks the arc
    stop_distance: float = 2.5           # free length below this commands a stop
    max_speed: float = 2.0
    w_traversability: float = 1.0
    w_clearance: float = 2.0
    w_curvature: float = 0.15
    w_goal: float = 0.3
    hysteresis: float = 0.08             # cost advantage needed to switch candidate
    max_curvature_rate: float = 0.5      # slew limit in 1/m per second
    obstacle_margin: float = 0.4


@dataclass(frozen=True)
class Obstacle:
    """Disc obstacle on the ground in the vehicle frame."""

    x: float
    y: float
    radius: float


@dataclass(frozen=True)
class PlanResult:
    curvature: float            # slew-limited command, 1/m
    target_curvature: float     # curvature of the selected candidate
    speed: float                # commanded speed, m/s
    selected: int               # index of the selected candidate
    costs: np.ndarray           # cost of every candidate (inf when blocked at once)
    free_lengths: np.ndarray    # collision-free arc length of every candidate
    switched: bool              # True when the selection changed this cycle


def arc_points(curvature: float, arc_lengths: np.ndarray) -> np.ndarray:
    """Points (x, y) on a constant-curvature arc starting at the origin heading along +x."""
    s = np.asarray(arc_lengths, dtype=np.float64)
    if abs(curvature) < 1e-9:
        return np.stack([s, np.zeros_like(s)], axis=1)
    return np.stack([np.sin(curvature * s) / curvature,
                     (1.0 - np.cos(curvature * s)) / curvature], axis=1)


@dataclass
class ArcRolloutPlanner:
    """Scores nine arcs on the traversability grid and commands curvature and speed.

    Hysteresis keeps the previous candidate unless another one is better by a margin,
    which prevents flip-flopping between two similar arcs. The published curvature is
    additionally slew-rate limited so the command is continuous.
    """

    spec: BevSpec
    config: PlannerConfig = field(default_factory=PlannerConfig)

    def __post_init__(self) -> None:
        cfg = self.config
        self.curvatures = np.linspace(-cfg.max_curvature, cfg.max_curvature, cfg.num_candidates)
        self._s = np.arange(cfg.step, cfg.horizon + 1e-9, cfg.step)
        self._arcs = [arc_points(k, self._s) for k in self.curvatures]
        self._cells = [self.spec.to_cell(a) for a in self._arcs]
        k = max(1, int(round(cfg.vehicle_width / self.spec.resolution)) | 1)
        self._footprint = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        self._selected: int | None = None
        self._command = 0.0

    def reset(self) -> None:
        self._selected = None
        self._command = 0.0

    def _sample(self, grid: np.ndarray, cells: np.ndarray) -> np.ndarray:
        """Grid value under each arc sample; samples outside the grid count as unknown."""
        rows, cols = grid.shape
        c = np.round(cells[:, 0]).astype(int)
        r = np.round(cells[:, 1]).astype(int)
        inside = (c >= 0) & (c < cols) & (r >= 0) & (r < rows)
        out = np.full(len(cells), 0.5)
        out[inside] = grid[r[inside], c[inside]]
        # The blind zone between the vehicle and the first BEV row is assumed free.
        out[cells[:, 1] > rows - 1] = 1.0
        return out

    def plan(self, traversability: np.ndarray, dt: float,
             obstacles: tuple[Obstacle, ...] = (), goal_curvature: float = 0.0) -> PlanResult:
        """Select an arc for the current traversability grid and obstacle list."""
        cfg = self.config
        # Erode by the vehicle footprint: a cell is as good as the worst cell under the body.
        grid = cv2.erode(traversability, self._footprint)
        n = len(self.curvatures)
        costs = np.full(n, np.inf)
        free = np.zeros(n)
        for i in range(n):
            trav = self._sample(grid, self._cells[i])
            blocked = trav < cfg.block_threshold
            for ob in obstacles:
                d = np.hypot(self._arcs[i][:, 0] - ob.x, self._arcs[i][:, 1] - ob.y)
                blocked |= d < ob.radius + 0.5 * cfg.vehicle_width + cfg.obstacle_margin
            first = int(np.argmax(blocked)) if blocked.any() else len(blocked)
            free[i] = self._s[first - 1] if first > 0 else 0.0
            if first == 0:
                continue
            costs[i] = (
                cfg.w_traversability * float(np.mean(1.0 - trav[:first]))
                + cfg.w_clearance * (1.0 - free[i] / cfg.horizon)
                + cfg.w_curvature * abs(self.curvatures[i]) / cfg.max_curvature
                + cfg.w_goal * abs(self.curvatures[i] - goal_curvature) / (2 * cfg.max_curvature)
            )
        best = int(np.argmin(costs))
        previous = self._selected
        if not np.isfinite(costs[best]):
            # Every arc is blocked right away: stop and hold the steering command.
            return PlanResult(self._command, self._command, 0.0, previous if previous is not None
                              else n // 2, costs, free, False)
        if previous is not None and np.isfinite(costs[previous]) \
                and costs[previous] <= costs[best] + cfg.hysteresis:
            best = previous
        self._selected = best
        target = float(self.curvatures[best])
        max_delta = cfg.max_curvature_rate * dt
        self._command += float(np.clip(target - self._command, -max_delta, max_delta))
        speed = 0.0 if free[best] < cfg.stop_distance else \
            cfg.max_speed * min(1.0, free[best] / cfg.horizon)
        return PlanResult(self._command, target, speed, best, costs, free,
                          previous is not None and best != previous)
