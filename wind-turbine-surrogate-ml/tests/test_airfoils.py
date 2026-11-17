import numpy as np
import pytest

from wtsurrogate.airfoils import (
    AIRFOIL_NAMES,
    get_airfoil,
    naca4_camber_slope,
    thin_airfoil_zero_lift_angle,
)


def test_zero_lift_angle_matches_closed_form_for_naca4412_camber():
    # thin-airfoil textbook value for a 4 percent camber at 40 percent chord: -4.15 deg
    assert np.degrees(thin_airfoil_zero_lift_angle(0.04, 0.4)) == pytest.approx(-4.15, abs=0.02)
    assert thin_airfoil_zero_lift_angle(0.0, 0.4) == 0.0


def test_camber_slope_is_zero_at_max_camber_position():
    assert naca4_camber_slope(np.array([0.4]), 0.04, 0.4)[0] == pytest.approx(0.0)
    assert naca4_camber_slope(np.array([0.1]), 0.04, 0.4)[0] > 0
    assert naca4_camber_slope(np.array([0.9]), 0.04, 0.4)[0] < 0


@pytest.mark.parametrize("name", AIRFOIL_NAMES)
def test_attached_flow_follows_thin_airfoil_slope(name):
    polar = get_airfoil(name)
    alpha = np.radians([0.0, 2.0, 4.0])
    cl, cd = polar.coefficients(alpha)
    slope = np.diff(cl) / np.diff(alpha)
    assert np.allclose(slope, polar.lift_slope, rtol=0.02)
    cl0, _ = polar.coefficients(np.array([polar.zero_lift_angle]))
    assert abs(cl0[0]) < 5e-3  # the stall blend leaks a little at the zero-lift angle
    assert np.all(cd > 0)


@pytest.mark.parametrize("name", AIRFOIL_NAMES)
def test_stall_limits_lift_and_raises_drag(name):
    polar = get_airfoil(name)
    alpha = np.radians(np.linspace(-20, 90, 221))
    cl, cd = polar.coefficients(alpha)
    assert 0.8 < cl.max() < 1.8
    assert np.degrees(alpha[cl.argmax()]) < np.degrees(polar.stall_angle) + 1
    assert cd[-1] == pytest.approx(polar.cd_max, rel=0.05)  # flat plate normal to the flow
    assert abs(cl[-1]) < 0.05


def test_camber_shifts_lift_and_drag_falls_with_reynolds():
    cambered, flatter = get_airfoil("naca4415"), get_airfoil("naca2215")
    assert cambered.zero_lift_angle < flatter.zero_lift_angle < 0
    _, cd_low = cambered.coefficients(np.array([0.05]), 1e6)
    _, cd_high = cambered.coefficients(np.array([0.05]), 1e7)
    assert cd_low[0] > cd_high[0]


def test_best_glide_and_unknown_airfoil():
    alpha, cl, glide = get_airfoil("naca4415").best_glide()
    assert 0 < np.degrees(alpha) < 10 and cl > 0.5 and glide > 50
    with pytest.raises(ValueError):
        get_airfoil("naca0012")
