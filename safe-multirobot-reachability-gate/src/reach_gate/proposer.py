"""Goal-seeking command proposers.

These are deliberately naive local controllers: they know the goal but nothing about
obstacles, corridors or other robots. They stand in for whatever produces candidate
motions in a real system (a planner, a learned policy, an operator). Safety comes only
from the gate that verifies their proposals.
"""

from __future__ import annotations

import heapq

import numpy as np

from .reachability import QuadrotorModel, RobotModel, UnicycleModel


def plan_route(start: np.ndarray, goal: np.ndarray,
               blocked: list[tuple[np.ndarray, np.ndarray]], bounds: tuple[np.ndarray, np.ndarray],
               inflate: float, resolution: float = 0.25) -> list[np.ndarray]:
    """Coarse 2D waypoint route with A* on an occupancy grid.

    ``blocked`` holds (lower, upper) xy rectangles, grown by ``inflate``. The route is a
    hint for the proposer, nothing more: it ignores dynamics, uncertainty and the other
    robots, and every command that follows it still has to pass the gate. Returns the
    waypoints after the start, ending with the goal, or just the goal if no route exists.
    """
    lo, hi = np.asarray(bounds[0], dtype=np.float64), np.asarray(bounds[1], dtype=np.float64)
    shape = np.maximum(np.ceil((hi - lo) / resolution).astype(int), 1)
    xs = lo[0] + (np.arange(shape[0]) + 0.5) * resolution
    ys = lo[1] + (np.arange(shape[1]) + 0.5) * resolution
    free = np.ones(shape, dtype=bool)
    edge = inflate - 1e-9
    free[(xs < lo[0] + edge) | (xs > hi[0] - edge), :] = False
    free[:, (ys < lo[1] + edge) | (ys > hi[1] - edge)] = False
    for b_lo, b_hi in blocked:
        in_x = (xs >= b_lo[0] - inflate) & (xs <= b_hi[0] + inflate)
        in_y = (ys >= b_lo[1] - inflate) & (ys <= b_hi[1] + inflate)
        free[np.ix_(in_x, in_y)] = False

    def nearest_free(p: np.ndarray) -> tuple[int, int] | None:
        cell = np.clip(((p - lo) / resolution).astype(int), 0, shape - 1)
        if free[cell[0], cell[1]]:
            return int(cell[0]), int(cell[1])
        idx = np.argwhere(free)
        if len(idx) == 0:
            return None
        best = idx[np.argmin(np.abs(idx - cell).sum(axis=1))]
        return int(best[0]), int(best[1])

    s, g = nearest_free(np.asarray(start[:2])), nearest_free(np.asarray(goal[:2]))
    if s is None or g is None:
        return [np.asarray(goal[:2], dtype=np.float64)]
    moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142)]
    best_cost = {s: 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    queue = [(0.0, s)]
    while queue:
        _, node = heapq.heappop(queue)
        if node == g:
            break
        for dx, dy, step in moves:
            nxt = (node[0] + dx, node[1] + dy)
            if not (0 <= nxt[0] < shape[0] and 0 <= nxt[1] < shape[1]) or not free[nxt]:
                continue
            if dx and dy and not (free[node[0] + dx, node[1]] and free[node[0], node[1] + dy]):
                continue  # do not cut corners diagonally
            cost = best_cost[node] + step
            if cost < best_cost.get(nxt, np.inf):
                best_cost[nxt] = cost
                parent[nxt] = node
                heapq.heappush(queue, (cost + float(np.hypot(g[0] - nxt[0], g[1] - nxt[1])), nxt))
    if g not in parent and g != s:
        return [np.asarray(goal[:2], dtype=np.float64)]
    cells = [g]
    while cells[-1] != s:
        cells.append(parent[cells[-1]])
    cells.reverse()

    def visible(a: tuple[int, int], b: tuple[int, int]) -> bool:
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2) + 1
        ii = np.round(np.linspace(a[0], b[0], n)).astype(int)
        jj = np.round(np.linspace(a[1], b[1], n)).astype(int)
        return bool(free[ii, jj].all())

    # Shortcut the cell path to a few straight legs.
    route, anchor = [], 0
    while anchor < len(cells) - 1:
        nxt = len(cells) - 1
        while nxt > anchor + 1 and not visible(cells[anchor], cells[nxt]):
            nxt -= 1
        route.append(np.array([xs[cells[nxt][0]], ys[cells[nxt][1]]]))
        anchor = nxt
    if route:
        route[-1] = np.asarray(goal[:2], dtype=np.float64)
    return route or [np.asarray(goal[:2], dtype=np.float64)]


def wrap_angle(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def _rank_by_progress(model: RobotModel, state: np.ndarray, goal: np.ndarray,
                      commands: list[np.ndarray], steps: int) -> list[np.ndarray]:
    """Order alternatives by the goal distance after a nominal rollout."""
    def end_distance(u: np.ndarray) -> float:
        x = state.copy()
        for _ in range(steps):
            x = model.step(x, u)
        return float(np.linalg.norm(model.position(x)[: goal.size] - goal))
    return sorted(commands, key=end_distance)


def propose_ground(model: UnicycleModel, state: np.ndarray, goal: np.ndarray,
                   speed_cap: float, steps: int = 10) -> list[np.ndarray]:
    """Primary command towards the goal followed by ranked alternatives."""
    delta = goal - state[:2]
    dist = float(np.linalg.norm(delta))
    err = wrap_angle(np.arctan2(delta[1], delta[0]) - state[2])
    v_max = min(speed_cap, model.max_speed)
    v = min(v_max, 1.2 * dist) * max(0.0, np.cos(err))
    w = float(np.clip(2.0 * err, -model.max_yaw_rate, model.max_yaw_rate))
    primary = np.array([v, w])
    v_alt = max(v, 0.6 * v_max)
    alternatives = []
    for dw in (0.5, -0.5, 1.0, -1.0):
        alternatives.append(model.clip(np.array([v_alt, w + dw])))
        alternatives.append(model.clip(np.array([0.5 * v_alt, w + dw])))
    alternatives += [np.array([0.0, 0.8]), np.array([0.0, -0.8])]
    return [primary, *_rank_by_progress(model, state, goal, alternatives, steps)]


def propose_aerial(model: QuadrotorModel, state: np.ndarray, goal: np.ndarray,
                   speed_cap: float, steps: int = 10) -> list[np.ndarray]:
    """Velocity command towards a 3D goal followed by ranked alternatives."""
    v_max = min(speed_cap, model.max_speed)
    desired = 1.0 * (goal - state[:3])
    n = float(np.linalg.norm(desired))
    primary = desired if n <= v_max else desired * (v_max / n)
    alternatives = [0.5 * primary]
    horizontal = primary[:2]
    for angle in np.deg2rad([45.0, -45.0, 90.0, -90.0]):
        c, s = np.cos(angle), np.sin(angle)
        rotated = np.array([c * horizontal[0] - s * horizontal[1],
                            s * horizontal[0] + c * horizontal[1], primary[2]])
        scale = max(np.linalg.norm(rotated), 0.6 * v_max) / max(np.linalg.norm(rotated), 1e-9)
        alternatives.append(model.clip(rotated * scale))
    return [primary, *_rank_by_progress(model, state, goal, alternatives, steps)]


def unsafe_candidates(model: RobotModel) -> list[np.ndarray]:
    """Full-speed commands in a fixed fan of directions, used for fault injection."""
    if isinstance(model, UnicycleModel):
        return [np.array([model.max_speed, w]) for w in (0.0, 0.6, -0.6, 1.2, -1.2)]
    out = []
    for angle in np.deg2rad(np.arange(0.0, 360.0, 45.0)):
        out.append(model.max_speed * np.array([np.cos(angle), np.sin(angle), 0.0]))
    out += [np.array([0.0, 0.0, -model.max_speed]), np.array([0.0, 0.0, model.max_speed])]
    return out
