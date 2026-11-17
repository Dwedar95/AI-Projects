import numpy as np
import pytest

from reach_gate.bus import MessageBus
from reach_gate.gate import Box, SafetyConstraints, SafetyGate
from reach_gate.reachability import QuadrotorModel, UnicycleModel, build_tube

WORKSPACE = Box((0.0, 0.0, 0.0), (20.0, 12.0, 4.0), "workspace")
WALL = Box((5.0, 4.0, 0.0), (6.0, 8.0, 3.0), "wall")


def ground_set(model, x, y, heading=0.0):
    return model.measurement_set(np.array([x, y, heading]))


def test_safe_command_is_published_and_unsafe_command_is_withheld():
    bus = MessageBus()
    gate = SafetyGate(bus)
    model = UnicycleModel()
    constraints = SafetyConstraints(WORKSPACE, obstacles=(WALL,))
    ok = gate.submit("ugv", model, ground_set(model, 2.0, 2.0), np.array([1.0, 0.0]), constraints)
    # 0.5 s at 1 m/s moves the 0.35 m footprint from x = 4.4 into the wall at x = 5.
    bad = gate.submit("ugv", model, ground_set(model, 4.4, 6.0), np.array([1.0, 0.0]), constraints)
    assert ok.safe and not bad.safe
    assert bad.violations[0].kind == "obstacle" and bad.violations[0].detail == "wall"
    assert bus.count("fleet/ugv/cmd/verified") == 1
    assert bus.count("fleet/ugv/cmd/withheld") == 1
    assert bus.log[-1].payload["reason"] == "obstacle"
    assert (gate.stats.verifications, gate.stats.published, gate.stats.withheld) == (2, 1, 1)
    assert gate.stats.by_kind == {"obstacle": 1}


def test_verify_has_no_side_effects():
    bus = MessageBus()
    gate = SafetyGate(bus)
    model = UnicycleModel()
    gate.verify("ugv", model, ground_set(model, 2.0, 2.0), np.array([1.0, 0.0]),
                SafetyConstraints(WORKSPACE))
    assert bus.log == [] and gate.stats.verifications == 0


def test_corridor_and_workspace_containment():
    gate = SafetyGate()
    model = UnicycleModel()
    corridor = Box((0.0, 1.0, 0.0), (20.0, 3.0, 4.0), "lane")
    constraints = SafetyConstraints(WORKSPACE, corridors={"ugv": corridor})
    along = gate.verify("ugv", model, ground_set(model, 2.0, 2.0), np.array([1.0, 0.0]),
                        constraints)
    across = gate.verify("ugv", model, ground_set(model, 2.0, 2.4, np.pi / 2),
                         np.array([1.0, 0.0]), constraints)
    assert along.safe
    assert not across.safe and across.violations[0].kind == "containment"
    # Another robot without a corridor may cross the lane.
    assert gate.verify("dog", model, ground_set(model, 2.0, 2.4, np.pi / 2),
                       np.array([1.0, 0.0]), constraints).safe
    leaving = gate.verify("ugv", model, ground_set(model, 19.3, 2.0), np.array([1.0, 0.0]),
                          constraints)
    assert not leaving.safe


def test_altitude_band_containment_for_quadrotor():
    gate = SafetyGate()
    model = QuadrotorModel()
    constraints = SafetyConstraints(WORKSPACE, altitude_bands={"uav": (1.5, 3.0)})
    x0 = model.measurement_set(np.array([5.0, 5.0, 2.2, 0.0, 0.0, 0.0]))
    assert gate.verify("uav", model, x0, np.array([1.0, 0.0, 0.0]), constraints).safe
    for vz in (-1.2, 1.2):
        verdict = gate.verify("uav", model, x0, np.array([0.0, 0.0, vz]), constraints)
        assert not verdict.safe and verdict.violations[0].kind == "containment"


def test_head_on_ground_robots_violate_separation():
    gate = SafetyGate()
    model = UnicycleModel()
    constraints = SafetyConstraints(WORKSPACE, min_separation=0.5)
    other = build_tube("dog", model, ground_set(model, 4.5, 2.0, np.pi),
                       [np.array([1.0, 0.0])] * 15)
    verdict = gate.verify("ugv", model, ground_set(model, 2.0, 2.0), np.array([1.0, 0.0]),
                          constraints, {"dog": other})
    assert not verdict.safe and verdict.violations[0].kind == "separation"
    assert verdict.violations[0].detail == "dog"
    far = build_tube("dog", model, ground_set(model, 4.5, 8.0, np.pi),
                     [np.array([1.0, 0.0])] * 15)
    assert gate.verify("ugv", model, ground_set(model, 2.0, 2.0), np.array([1.0, 0.0]),
                       constraints, {"dog": far}).safe


def test_separation_is_three_dimensional():
    """The quadrotor may pass over a ground robot only with enough vertical clearance."""
    gate = SafetyGate()
    ugv, uav = UnicycleModel(), QuadrotorModel()
    constraints = SafetyConstraints(WORKSPACE, min_separation=0.5)
    parked = build_tube("ugv", ugv, ground_set(ugv, 5.0, 5.0), [np.zeros(2)] * 15)
    command = np.array([1.2, 0.0, 0.0])
    high = uav.measurement_set(np.array([4.0, 5.0, 2.5, 1.2, 0.0, 0.0]))
    low = uav.measurement_set(np.array([4.0, 5.0, 1.0, 1.2, 0.0, 0.0]))
    assert gate.verify("uav", uav, high, command, constraints, {"ugv": parked}).safe
    verdict = gate.verify("uav", uav, low, command, constraints, {"ugv": parked})
    assert not verdict.safe and verdict.violations[0].kind == "separation"


def test_speed_limit_is_enforced_on_the_command():
    gate = SafetyGate()
    model = UnicycleModel()
    constraints = SafetyConstraints(WORKSPACE, speed_limits={"ugv": 0.5})
    x0 = ground_set(model, 2.0, 2.0)
    fast = gate.verify("ugv", model, x0, np.array([0.9, 0.0]), constraints)
    assert not fast.safe and fast.violations[0].kind == "speed_limit"
    assert gate.verify("ugv", model, x0, np.array([0.5, 0.0]), constraints).safe


def test_contingency_tail_is_part_of_the_check():
    """A fast quadrotor command is withheld if it cannot stop before the wall."""
    model = QuadrotorModel()
    constraints = SafetyConstraints(WORKSPACE, obstacles=(WALL,))
    x0 = model.measurement_set(np.array([3.6, 6.0, 2.0, 1.4, 0.0, 0.0]))
    command = np.array([1.4, 0.0, 0.0])
    short = SafetyGate(horizon=4, commit_steps=4).verify("uav", model, x0, command, constraints)
    full = SafetyGate(horizon=15, commit_steps=5).verify("uav", model, x0, command, constraints)
    assert short.safe          # the wall is not reached within 0.4 s
    assert not full.safe       # but it is reached while braking afterwards
    with pytest.raises(ValueError):
        SafetyGate(horizon=5, commit_steps=6)


@pytest.mark.parametrize("seed", range(5))
def test_soundness_unsafe_commands_are_always_withheld(seed):
    """Property: if any simulated trajectory violates a constraint, the gate said no."""
    rng = np.random.default_rng(seed)
    gate = SafetyGate()
    model = UnicycleModel()
    constraints = SafetyConstraints(
        WORKSPACE, obstacles=(WALL, Box((9.0, 1.0, 0.0), (10.0, 3.0, 1.0), "crate")))
    lo_box, hi_box = constraints.containment_box("ugv")
    half = model.body_half_extent()
    unsafe_seen = safe_seen = 0
    for _ in range(40):
        measured = np.array([rng.uniform(0.5, 12.0), rng.uniform(0.5, 11.5),
                             rng.uniform(-np.pi, np.pi)])
        command = np.array([rng.uniform(0.2, 1.0), rng.uniform(-1.2, 1.2)])
        x0 = model.measurement_set(measured)
        verdict = gate.verify("ugv", model, x0, command, constraints)
        violated = False
        for start in x0.sample(rng, 8, boundary=True):
            x = start
            for k in range(gate.horizon):
                u = command if k < gate.commit_steps else model.stop_command()
                x = model.simulate_step(x, u, rng)
                p = model.position(x)
                if np.any(p - half < lo_box) or np.any(p + half > hi_box) or any(
                        ob.intersects_box(p - half, p + half) for ob in constraints.obstacles):
                    violated = True
        if violated:
            unsafe_seen += 1
            assert not verdict.safe
        safe_seen += int(verdict.safe)
    assert unsafe_seen > 0 and safe_seen > 0  # the test exercises both outcomes
