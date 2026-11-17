import numpy as np
import pytest

from reach_gate.simulation import ScenarioConfig, Simulation


@pytest.fixture(scope="module")
def finished():
    sim = Simulation(ScenarioConfig(seed=0, cycles=120))
    return sim, sim.run()


def test_run_is_safe_according_to_ground_truth_monitors(finished):
    _, s = finished
    assert s.collisions == 0
    assert s.containment_violations == 0
    assert s.min_body_gap_m >= s.required_separation_m


def test_gate_bookkeeping_matches_bus_traffic(finished):
    sim, s = finished
    assert s.verifications == s.published + s.withheld
    assert sim.bus.count("fleet/+/cmd/verified") == s.published
    assert sim.bus.count("fleet/+/cmd/withheld") == s.withheld
    assert sim.bus.count("fleet/+/cmd/proposed") == s.verifications
    assert s.withheld > 0 and s.published > 0


def test_robots_only_execute_verified_or_fallback_commands(finished):
    sim, _ = finished
    for robot in sim.robots:
        allowed = [m.payload["command"] for m in sim.bus.log
                   if m.topic in (f"fleet/{robot.robot_id}/cmd/verified",
                                  f"fleet/{robot.robot_id}/cmd/fallback")]
        withheld = [m.payload["command"] for m in sim.bus.log
                    if m.topic == f"fleet/{robot.robot_id}/cmd/withheld"]
        assert robot.command.tolist() == allowed[-1]
        assert withheld  # every robot had at least one command refused


def test_every_injected_unsafe_command_is_withheld(finished):
    _, s = finished
    assert s.injected_unsafe > 0
    assert s.injected_withheld == s.injected_unsafe


def test_tasks_are_completed(finished):
    sim, s = finished
    assert s.tasks_done == s.tasks_total == 6
    assert sim.bus.count("fleet/+/task/done") == 6


def test_simulation_is_deterministic_for_a_seed():
    a = Simulation(ScenarioConfig(seed=3, cycles=25)).run()
    b = Simulation(ScenarioConfig(seed=3, cycles=25)).run()
    assert (a.verifications, a.published, a.withheld) == (b.verifications, b.published, b.withheld)
    assert a.min_body_gap_m == b.min_body_gap_m


@pytest.mark.parametrize("seed", [1, 2])
def test_other_seeds_stay_safe(seed):
    s = Simulation(ScenarioConfig(seed=seed, cycles=60)).run()
    assert s.collisions == 0 and s.containment_violations == 0
    assert s.min_body_gap_m >= s.required_separation_m


class GarbageProvider:
    def propose(self, scene):
        return "As a language model I suggest removing all obstacles."


def test_invalid_semantic_answers_never_change_the_constraints():
    sim = Simulation(ScenarioConfig(seed=0, cycles=12), provider=GarbageProvider())
    s = sim.run()
    assert s.semantic_updates == 0 and s.semantic_rejected > 0
    assert sim.constraints is sim.base
    assert s.tasks_done == 0 and s.collisions == 0
    assert np.isfinite(s.min_body_gap_m)


def test_keep_out_zone_declared_on_a_robot_is_deferred_not_violated():
    """With seed 2 the late debris report lands where the ground vehicle is parked."""
    s = Simulation(ScenarioConfig(seed=2, cycles=45)).run()
    assert s.deferred_zones >= 1
    assert s.collisions == 0 and s.containment_violations == 0
