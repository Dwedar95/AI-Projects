import numpy as np
import pytest

from wtsurrogate.bem import prandtl_loss, solve_bem
from wtsurrogate.rotor import AIR_DENSITY, reference_rotor

BETZ = 16.0 / 27.0


def test_reference_rotor_geometry(rotor):
    assert rotor.r[0] > rotor.root_radius and rotor.r[-1] < rotor.radius
    assert rotor.chord.max() <= 6.2 + 1e-12
    assert np.all(np.diff(rotor.twist) < 0)  # twist decreases towards the tip
    assert 0.02 < rotor.solidity < 0.08
    scaled = rotor.scaled(1.1, 0.9)
    assert np.allclose(scaled.chord, 1.1 * rotor.chord)
    assert np.allclose(scaled.twist, 0.9 * rotor.twist)
    with pytest.raises(ValueError):
        reference_rotor(n_elements=2)


def test_prandtl_loss_limits():
    r = np.array([20.0, 50.0, 88.9])
    loss = prandtl_loss(np.full(3, 0.1), r, 89.0, 13.35, 3)
    assert loss[1] == pytest.approx(1.0, abs=1e-3)
    assert loss[2] < 0.5
    assert np.all((loss > 0) & (loss <= 1))


def test_cp_is_physical_and_peaks_near_design_tsr(rotor):
    tsr = np.linspace(4.0, 12.0, 33)
    res = solve_bem(rotor, 9.0, tsr, 0.0)
    assert res.converged.all()
    assert 0.42 < res.cp.max() < BETZ
    assert abs(tsr[res.cp.argmax()] - rotor.design_tsr) < 1.5
    assert np.all(np.diff(res.ct) > 0)  # thrust coefficient grows with TSR at fixed pitch


def test_solution_satisfies_momentum_balance(rotor):
    """Blade element thrust equals momentum thrust on every element below the Buhl range."""
    res = solve_bem(rotor, 9.0, 6.0, np.radians(3.0))
    a, phi = res.axial_induction[0], res.phi[0]
    loss = prandtl_loss(phi, rotor.r, rotor.radius, rotor.root_radius, rotor.n_blades)
    momentum = 4 * np.pi * rotor.r * AIR_DENSITY * 9.0**2 * a * (1 - a) * loss
    element = rotor.n_blades * res.normal_load[0]
    light = a < 0.39
    assert light.sum() >= 25 and np.all(a > 0)
    assert np.allclose(element[light], momentum[light], rtol=1e-4)
    assert res.thrust[0] == pytest.approx(element.sum() * rotor.dr)


def test_power_matches_cp_definition_and_torque(rotor):
    res = solve_bem(rotor, 10.0, 7.0, np.radians(1.0))
    omega = 7.0 * 10.0 / rotor.radius
    assert res.power[0] == pytest.approx(res.torque[0] * omega)
    expected = res.cp[0] * 0.5 * AIR_DENSITY * rotor.swept_area * 10.0**3
    assert res.power[0] == pytest.approx(expected)
    assert 5e6 < res.power[0] < 9e6


def test_batch_equals_individual_calls(rotor, rng):
    u, tsr = rng.uniform(5, 12, 6), rng.uniform(5, 10, 6)
    pitch = np.radians(rng.uniform(-2, 6, 6))
    cs, ts = rng.uniform(0.8, 1.2, 6), rng.uniform(0.8, 1.2, 6)
    batch = solve_bem(rotor, u, tsr, pitch, cs, ts)
    for i in range(6):
        single = solve_bem(rotor, u[i], tsr[i], pitch[i], cs[i], ts[i])
        assert single.cp[0] == pytest.approx(batch.cp[i], rel=1e-9)


def test_pitch_to_feather_sheds_power_and_scaling_matches_scaled_rotor(rotor):
    res = solve_bem(rotor, 15.0, 6.0, np.radians([0.0, 10.0, 20.0]))
    assert res.power[0] > res.power[1] > res.power[2]
    a = solve_bem(rotor, 9.0, 8.0, 0.0, chord_scale=1.15, twist_scale=0.9)
    b = solve_bem(rotor.scaled(1.15, 0.9), 9.0, 8.0, 0.0)
    assert a.cp[0] == pytest.approx(b.cp[0])


def test_reynolds_effect_is_small_but_present(rotor):
    res = solve_bem(rotor, np.array([5.0, 12.0]), 7.5, 0.0)
    assert 0 < res.cp[1] - res.cp[0] < 0.01


def test_invalid_inputs(rotor):
    with pytest.raises(ValueError):
        solve_bem(rotor, -1.0, 7.0)
    with pytest.raises(ValueError):
        solve_bem(rotor, 9.0, 0.0)
