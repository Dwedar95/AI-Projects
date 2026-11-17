import json

import numpy as np
import pytest

from reach_gate.bus import MessageBus, topic_matches
from reach_gate.gate import Box, SafetyConstraints
from reach_gate.proposer import plan_route, propose_aerial, propose_ground
from reach_gate.reachability import QuadrotorModel, UnicycleModel
from reach_gate.semantic import (
    Directive,
    KeepOutZone,
    MockSemanticProvider,
    Observation,
    RobotInfo,
    SceneReport,
    SemanticPlan,
    Task,
    merge_constraints,
    parse_plan,
)

BASE = SafetyConstraints(
    Box((0.0, 0.0, 0.0), (20.0, 12.0, 4.0)), obstacles=(Box((6, 4, 0), (8, 7, 3), "rack"),),
    altitude_bands={"uav": (1.5, 3.2)}, speed_limits={"dog": 0.6}, min_separation=0.5)
SPEEDS = {"uav": 1.5, "ugv": 1.2, "dog": 0.8}


def scene(tasks=None, observations=None, busy=()):
    robots = [
        RobotInfo(robot_id="uav", kind="aerial", capabilities=["aerial"],
                  position=(1.0, 6.0, 2.5), busy="uav" in busy),
        RobotInfo(robot_id="ugv", kind="ground", capabilities=["ground", "carry"],
                  position=(1.0, 3.0, 0.3), busy="ugv" in busy),
        RobotInfo(robot_id="dog", kind="ground", capabilities=["ground", "legged"],
                  position=(1.0, 9.0, 0.25), busy="dog" in busy),
    ]
    return SceneReport(robots=robots, tasks=tasks or [], observations=observations or [])


@pytest.mark.parametrize("pattern,topic,expected", [
    ("fleet/uav/state", "fleet/uav/state", True),
    ("fleet/+/state", "fleet/ugv/state", True),
    ("fleet/+/state", "fleet/ugv/cmd/verified", False),
    ("fleet/#", "fleet/ugv/cmd/verified", True),
    ("fleet/+/cmd/#", "fleet/dog/cmd/withheld", True),
    ("fleet/uav", "fleet/uav/state", False),
    ("#", "anything/at/all", True),
    ("fleet/uav/state/extra", "fleet/uav/state", False),
])
def test_topic_matching(pattern, topic, expected):
    assert topic_matches(pattern, topic) is expected


def test_bus_delivery_order_retained_messages_and_counts():
    bus = MessageBus()
    seen = []
    bus.subscribe("fleet/+/state", lambda m: seen.append(("wild", m.topic)))
    bus.subscribe("fleet/uav/state", lambda m: seen.append(("exact", m.topic)))
    bus.publish("fleet/uav/state", {"x": 1})
    bus.publish("fleet/ugv/state", {"x": 2})
    assert seen == [("wild", "fleet/uav/state"), ("exact", "fleet/uav/state"),
                    ("wild", "fleet/ugv/state")]
    bus.publish("fleet/semantic/plan", {"v": 1}, retain=True)
    bus.publish("fleet/semantic/plan", {"v": 2}, retain=True)
    late = []
    bus.subscribe("fleet/semantic/#", lambda m: late.append(m.payload["v"]))
    assert late == [2]  # a late subscriber gets only the last retained message
    assert bus.count("fleet/+/state") == 2 and [m.seq for m in bus.log] == [1, 2, 3, 4]
    with pytest.raises(ValueError):
        bus.publish("fleet/+/state", {})
    with pytest.raises(ValueError):
        bus.subscribe("fleet/#/state", lambda m: None)


def test_mock_provider_is_deterministic_and_returns_valid_json():
    s = scene(tasks=[Task(task_id="t1", capability="carry", location=(15.0, 5.0))],
              observations=[Observation(label="person", position=(3.0, 3.0), confidence=0.9)])
    provider = MockSemanticProvider(SPEEDS)
    text = provider.propose(s)
    assert text == provider.propose(s)
    plan = parse_plan(text)
    assert plan is not None and json.loads(text)["keep_out"][0]["label"] == "person"
    zone = plan.keep_out[0]
    assert zone.half_size == (1.3, 1.3) and zone.height == 4.0
    by_id = {d.robot_id: d for d in plan.directives}
    assert by_id["ugv"].max_speed == pytest.approx(0.6)   # 2.0 m from the person: half speed
    assert by_id["uav"].altitude_band == (2.0, 3.0)
    assert plan.min_separation == 0.8


def test_mock_allocation_respects_capabilities_and_idle_state():
    tasks = [Task(task_id="fly", capability="aerial", location=(1.0, 9.0)),
             Task(task_id="haul", capability="carry", location=(1.0, 9.5)),
             Task(task_id="walk", capability="legged", location=(1.0, 3.0))]
    plan = parse_plan(MockSemanticProvider(SPEEDS).propose(scene(tasks=tasks)))
    assigned = {d.robot_id: d.task_id for d in plan.directives}
    # Distance alone would give "haul" to the dog and "walk" to the ugv.
    assert assigned == {"uav": "fly", "ugv": "haul", "dog": "walk"}
    plan = parse_plan(MockSemanticProvider(SPEEDS).propose(scene(tasks=tasks, busy=("ugv",))))
    assert {d.robot_id: d.task_id for d in plan.directives}["ugv"] is None


def test_low_confidence_observations_are_ignored():
    obs = [Observation(label="person", position=(3.0, 3.0), confidence=0.1)]
    plan = parse_plan(MockSemanticProvider(SPEEDS).propose(scene(observations=obs)))
    assert plan.keep_out == [] and plan.min_separation is None


@pytest.mark.parametrize("text", [
    "I think the robots should be careful.",
    "{not json}",
    '{"keep_out": [{"label": "x", "center": [1, 2], "half_size": [1, 1], "height": -3}]}',
    '{"min_separation": -1}',
    '{"directives": [{"robot_id": "uav", "max_speed": 0}]}',
    '{"keep_out": "everywhere"}',
])
def test_invalid_model_answers_are_rejected(text):
    assert parse_plan(text) is None


def test_parse_plan_accepts_json_wrapped_in_prose():
    text = 'Here is the plan:\n```json\n{"rationale": "ok", "min_separation": 1.0}\n```'
    plan = parse_plan(text)
    assert plan is not None and plan.min_separation == 1.0


def test_merge_can_only_tighten_constraints():
    """Property: whatever the plan says, the merged constraints are never looser."""
    rng = np.random.default_rng(0)
    for _ in range(100):
        plan = SemanticPlan(
            keep_out=[KeepOutZone(label="z", center=tuple(rng.uniform(0, 20, 2)),
                                  half_size=tuple(rng.uniform(0.1, 2, 2)),
                                  height=float(rng.uniform(0.1, 5)))
                      for _ in range(rng.integers(0, 3))],
            directives=[
                Directive(robot_id="uav", altitude_band=tuple(rng.uniform(0, 6, 2)),
                          max_speed=float(rng.uniform(0.1, 5))),
                Directive(robot_id="dog", max_speed=float(rng.uniform(0.1, 5))),
                Directive(robot_id="ugv", altitude_band=(0.0, 9.0)),
            ],
            min_separation=float(rng.uniform(0.01, 3)) if rng.random() < 0.7 else None)
        merged = merge_constraints(BASE, plan, SPEEDS)
        assert merged.workspace == BASE.workspace and merged.corridors == BASE.corridors
        assert set(BASE.obstacles) <= set(merged.obstacles)
        assert merged.min_separation >= BASE.min_separation
        lo, hi = merged.altitude_bands["uav"]
        assert lo >= 1.5 and hi <= 3.2 and lo < hi
        assert "ugv" not in merged.altitude_bands
        assert merged.speed_limits["dog"] <= 0.6
        assert merged.speed_limits["uav"] <= SPEEDS["uav"]


def test_route_hint_goes_around_a_blocking_box():
    blocked = [(np.array([6.0, 4.0]), np.array([8.0, 7.5]))]
    bounds = (np.array([0.0, 0.0]), np.array([20.0, 12.0]))
    route = plan_route(np.array([2.0, 6.0]), np.array([12.0, 6.0]), blocked, bounds, inflate=0.8)
    assert len(route) >= 2 and np.allclose(route[-1], [12.0, 6.0])
    points = np.vstack([[2.0, 6.0], *route])
    for a, b in zip(points[:-1], points[1:], strict=True):
        for lam in np.linspace(0, 1, 50):
            p = (1 - lam) * a + lam * b
            assert not (6.0 - 0.5 < p[0] < 8.0 + 0.5 and 4.0 - 0.5 < p[1] < 7.5 + 0.5)
    direct = plan_route(np.array([2.0, 2.0]), np.array([12.0, 2.0]), blocked, bounds, 0.8)
    assert len(direct) == 1


def test_proposers_put_the_goal_seeking_command_first():
    ugv = UnicycleModel()
    cands = propose_ground(ugv, np.array([0.0, 0.0, 0.0]), np.array([5.0, 0.0]), speed_cap=0.7)
    assert np.allclose(cands[0], [0.7, 0.0]) and len(cands) == 11
    turn = propose_ground(ugv, np.array([0.0, 0.0, 0.0]), np.array([0.0, 5.0]), speed_cap=1.0)
    assert turn[0][0] == pytest.approx(0.0, abs=1e-9) and turn[0][1] > 0.0
    uav = QuadrotorModel()
    air = propose_aerial(uav, np.array([0.0, 0.0, 2.0, 0, 0, 0]), np.array([10.0, 0.0, 2.0]), 1.0)
    assert np.allclose(air[0], [1.0, 0.0, 0.0])
    assert all(np.linalg.norm(c) <= uav.max_speed + 1e-9 for c in air)
