import numpy as np
import pytest

from energy_iot.simulators import Battery, PVArray, PVModule, WindTurbine, simulate_weather


@pytest.fixture(scope="module")
def module() -> PVModule:
    return PVModule()


def test_iv_curve_endpoints(module):
    isc = float(module.current(0.0, 1000.0, 25.0))
    assert isc == pytest.approx(module.iph_ref_a, rel=0.01)
    voc = module.open_circuit_voltage(1000.0, 25.0)
    assert voc == pytest.approx(module.voc_ref_v, abs=0.05)
    assert abs(float(module.current(voc, 1000.0, 25.0))) < 1e-6


def test_current_satisfies_single_diode_equation(module):
    v, g, t = 28.0, 700.0, 40.0
    i = float(module.current(v, g, t))
    a = float(module._a(t))
    rhs = (
        float(module.photocurrent(g, t))
        - float(module.saturation_current(t)) * np.expm1((v + i * module.rs_ohm) / a)
        - (v + i * module.rs_ohm) / module.rsh_ohm
    )
    assert i == pytest.approx(rhs, abs=1e-8)


def test_current_decreases_with_voltage(module):
    v = np.linspace(0.0, 37.0, 200)
    i = module.current(v, 800.0, 30.0)
    assert np.all(np.diff(i) < 0)


def test_mpp_is_the_maximum_of_a_dense_scan(module):
    for g, t in [(1000.0, 25.0), (400.0, 45.0), (120.0, 10.0)]:
        mpp = module.mpp(g, t)
        v = np.linspace(0.0, module.open_circuit_voltage(g, t), 4000)
        p = module.power(v, g, t)
        assert mpp.power_w >= p.max() - 1e-6
        assert mpp.voltage_v == pytest.approx(v[np.argmax(p)], abs=0.02)


def test_mpp_batch_agrees_with_scalar_optimiser(module):
    g = np.array([100.0, 350.0, 800.0, 1050.0])
    t = np.array([5.0, 25.0, 50.0, 65.0])
    v_b, p_b = module.mpp_batch(g, t)
    for k in range(len(g)):
        ref = module.mpp(float(g[k]), float(t[k]))
        assert p_b[k] == pytest.approx(ref.power_w, rel=1e-6)
        assert v_b[k] == pytest.approx(ref.voltage_v, abs=5e-3)


def test_mpp_physics_trends(module):
    assert module.mpp(1000.0, 25.0).power_w > module.mpp(500.0, 25.0).power_w
    # Higher cell temperature lowers MPP voltage and power.
    hot, cold = module.mpp(1000.0, 60.0), module.mpp(1000.0, 25.0)
    assert hot.voltage_v < cold.voltage_v and hot.power_w < cold.power_w
    v, p = module.mpp_batch(np.array([0.0]), np.array([25.0]))
    assert v[0] == 0.0 and p[0] == 0.0


def test_pv_array_scales_module_power(module):
    array = PVArray(module, n_series=4, n_parallel=3, inverter_efficiency=1.0)
    assert array.rated_power_w == pytest.approx(12 * module.mpp(1000.0, 25.0).power_w)
    point = array.operating_point(np.array([600.0]), np.array([20.0]))
    assert point["voltage_v"][0] * point["current_a"][0] == pytest.approx(point["power_w"][0])


def test_wind_power_curve():
    turbine = WindTurbine()
    p = turbine.power(np.array([0.0, 2.9, 3.0, 8.0, 12.0, 20.0, 25.0, 30.0]))
    assert p[0] == 0 and p[1] == 0 and p[2] == 0
    assert 0 < p[3] < turbine.rated_power_w
    assert p[4] == turbine.rated_power_w and p[5] == turbine.rated_power_w
    assert p[6] == 0 and p[7] == 0


def test_battery_respects_limits_and_efficiency():
    b = Battery(capacity_wh=1000.0, max_charge_w=500.0, max_discharge_w=500.0, soc=0.5)
    assert b.step(2000.0, 3600.0) == pytest.approx((0.95 - 0.5) * 1000.0 / 0.95)
    assert b.soc == pytest.approx(0.95)
    assert b.step(100.0, 3600.0) == pytest.approx(0.0)
    assert b.step(-300.0, 1800.0) == pytest.approx(-300.0)
    assert b.soc == pytest.approx(0.95 - 150.0 / 0.95 / 1000.0)
    b.step(-500.0, 36000.0)
    assert b.soc == pytest.approx(b.soc_min)
    assert b.step(-100.0, 3600.0) == pytest.approx(0.0)


def test_battery_round_trip_loses_energy():
    b = Battery(capacity_wh=1000.0, soc=0.2)
    charged = b.step(400.0, 3600.0)
    recovered = 0.0
    for _ in range(10):
        recovered += -b.step(-400.0, 3600.0, floor_soc=0.2)
    assert recovered == pytest.approx(charged * 0.95 * 0.95)


def test_weather_is_reproducible_and_plausible():
    a = simulate_weather(96 * 3, 900.0, seed=5)
    b = simulate_weather(96 * 3, 900.0, seed=5)
    c = simulate_weather(96 * 3, 900.0, seed=6)
    assert np.array_equal(a.irradiance_wm2, b.irradiance_wm2)
    assert not np.array_equal(a.irradiance_wm2, c.irradiance_wm2)
    hour = (a.ts / 3600.0) % 24.0
    assert np.all(a.irradiance_wm2[(hour < 5) | (hour > 21)] == 0.0)
    assert a.irradiance_wm2.max() <= 950.0 and a.irradiance_wm2.min() >= 0.0
    assert a.wind_speed_ms.min() >= 0.0 and a.load_w.min() >= 50.0
