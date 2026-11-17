import numpy as np
import pytest

from reach_gate.reachability import (
    QuadrotorModel,
    UnicycleModel,
    build_tube,
    contingency_inputs,
    shift_tube,
)
from reach_gate.zonotope import Zonotope


def rollout(model, x0, inputs, rng, extreme=False):
    """True plant trajectory. With ``extreme`` the errors sit on the bounds."""
    xs = [np.asarray(x0, dtype=np.float64)]
    for u in inputs:
        if extreme and isinstance(model, UnicycleModel) and np.any(u):
            du = rng.choice([-1.0, 1.0], 2) * np.asarray(model.input_error)
            w = rng.choice([-1.0, 1.0], 3) * np.asarray(model.disturbance)
            xs.append(model.step(xs[-1], np.asarray(u) + du) + w)
        elif extreme and isinstance(model, QuadrotorModel):
            d = rng.choice([-1.0, 1.0], 3) * np.asarray(model.disturbance)
            xs.append(model.step(xs[-1], u) + model.E @ d)
        else:
            xs.append(model.simulate_step(xs[-1], u, rng))
    return xs


@pytest.mark.parametrize("seed", range(6))
def test_unicycle_sampled_trajectories_stay_inside_reachable_sets(seed):
    rng = np.random.default_rng(seed)
    model = UnicycleModel()
    measured = np.array([rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-np.pi, np.pi)])
    command = np.array([rng.uniform(-1.0, 1.0), rng.uniform(-1.2, 1.2)])
    inputs = contingency_inputs(model, command, horizon=15, commit_steps=10)
    x0 = model.measurement_set(measured)
    sets = model.reach(x0, inputs)
    assert len(sets) == 16
    starts = np.vstack([x0.sample(rng, 6), x0.sample(rng, 6, boundary=True)])
    for i, start in enumerate(starts):
        for k, x in enumerate(rollout(model, start, inputs, rng, extreme=i % 2 == 0)):
            assert sets[k].contains_point(x), f"escaped at step {k}"


def test_unicycle_remainder_is_needed_for_large_heading_uncertainty():
    """With a wide heading set the linear part alone does not cover the true arc."""
    model = UnicycleModel(state_noise=(0.0, 0.0, 0.6), input_error=(0.0, 0.0),
                          disturbance=(0.0, 0.0, 0.0))
    x0 = model.measurement_set(np.zeros(3))
    inputs = [np.array([1.0, 0.0])] * 10
    sets = model.reach(x0, inputs)
    # Heading at the bound: the robot moves along a 0.6 rad line, cos(0.6) < 1.
    x = np.array([0.0, 0.0, 0.6])
    for k, u in enumerate(inputs):
        x = model.step(x, u)
        assert sets[k + 1].contains_point(x)
    assert model.lagrange_remainder(1.0, 0.6, 0.0) > 0.01
    assert model.lagrange_remainder(1.0, 0.0, 0.0) == 0.0


def test_unicycle_jacobians_match_finite_differences():
    model = UnicycleModel()
    x, u = np.array([0.3, -0.2, 0.7]), np.array([0.8, -0.4])
    a, b = model.jacobians(x, u)
    eps = 1e-6
    for i in range(3):
        d = np.zeros(3)
        d[i] = eps
        assert np.allclose((model.step(x + d, u) - model.step(x - d, u)) / (2 * eps), a[:, i],
                           atol=1e-6)
    for i in range(2):
        d = np.zeros(2)
        d[i] = eps
        assert np.allclose((model.step(x, u + d) - model.step(x, u - d)) / (2 * eps), b[:, i],
                           atol=1e-6)


def test_holding_ground_robot_is_invariant():
    model = UnicycleModel()
    x0 = model.measurement_set(np.array([1.0, 2.0, 0.3]))
    sets = model.reach(x0, [np.zeros(2)] * 20)
    assert all(s is x0 for s in sets)
    rng = np.random.default_rng(0)
    x = np.array([1.0, 2.0, 0.3])
    assert np.array_equal(model.simulate_step(x, np.zeros(2), rng), x)


@pytest.mark.parametrize("seed", range(4))
def test_quadrotor_sampled_trajectories_stay_inside_reachable_sets(seed):
    rng = np.random.default_rng(100 + seed)
    model = QuadrotorModel()
    measured = np.concatenate([rng.uniform(-2, 2, 3), rng.uniform(-1, 1, 3)])
    command = model.clip(rng.uniform(-1.5, 1.5, 3))
    inputs = contingency_inputs(model, command, horizon=15, commit_steps=5)
    x0 = model.measurement_set(measured)
    sets = model.reach(x0, inputs)
    starts = np.vstack([x0.sample(rng, 5), x0.sample(rng, 5, boundary=True)])
    for i, start in enumerate(starts):
        for k, x in enumerate(rollout(model, start, inputs, rng, extreme=i % 2 == 0)):
            assert sets[k].contains_point(x), f"escaped at step {k}"


def test_quadrotor_reach_centre_follows_nominal_and_velocity_converges():
    model = QuadrotorModel()
    x = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    u = np.array([1.0, 0.0, 0.0])
    sets = model.reach(Zonotope.point(x), [u] * 30)
    for k in range(30):
        x = model.step(x, u)
        assert np.allclose(sets[k + 1].center, x)
    assert abs(x[3] - 1.0) < 1e-3  # first-order velocity loop settles at the command
    assert np.linalg.norm(model.clip(np.array([3.0, 4.0, 0.0]))) == pytest.approx(1.5)


def test_reach_set_grows_with_disturbance_and_time():
    quiet = UnicycleModel(disturbance=(0.0, 0.0, 0.0), input_error=(0.0, 0.0))
    noisy = UnicycleModel()
    x0 = Zonotope.point(np.zeros(3))
    u = [np.array([0.8, 0.3])] * 15
    r_quiet = [s.radius() for s in quiet.reach(x0, u)]
    r_noisy = [s.radius() for s in noisy.reach(x0, u)]
    assert np.allclose(r_quiet[-1], 0.0)
    assert np.all(r_noisy[-1] > r_noisy[5]) and np.all(r_noisy[5] > 0.0)


def test_swept_occupancy_contains_body_between_samples():
    rng = np.random.default_rng(7)
    model = UnicycleModel()
    x0 = model.measurement_set(np.array([0.0, 0.0, 0.4]))
    inputs = [np.array([1.0, 0.8])] * 12
    tube = build_tube("r", model, x0, inputs)
    assert tube.steps == 12
    half = model.body_half_extent()
    xs = rollout(model, x0.sample(rng, 1)[0], inputs, rng)
    for k in range(12):
        for lam in (0.0, 0.5, 1.0):
            centre = (1 - lam) * model.position(xs[k]) + lam * model.position(xs[k + 1])
            for corner in ((1, 1, 1), (-1, 1, -1), (1, -1, -1), (-1, -1, 1)):
                assert tube.occupancy[k].contains_point(centre + half * np.array(corner))


def test_shifted_tube_still_contains_the_continued_trajectory():
    rng = np.random.default_rng(8)
    model = QuadrotorModel()
    x0 = model.measurement_set(np.array([0.0, 0.0, 2.5, 0.5, 0.0, 0.0]))
    inputs = contingency_inputs(model, np.array([1.0, 0.5, 0.0]), 15, 5)
    tube = build_tube("uav", model, x0, inputs)
    shifted = shift_tube(tube, model, 5)
    assert shifted.steps == tube.steps and len(shifted.states) == len(tube.states)
    xs = rollout(model, x0.sample(rng, 1)[0], inputs + [model.stop_command()] * 5, rng)
    for k in range(16):
        assert shifted.states[k].contains_point(xs[k + 5])
