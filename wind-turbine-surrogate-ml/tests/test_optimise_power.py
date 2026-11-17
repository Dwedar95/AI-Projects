import numpy as np
import pytest

from wtsurrogate.optimise import maximise_cp, raw_row, throughput
from wtsurrogate.power import (
    HOURS_PER_YEAR,
    TurbineLimits,
    annual_energy_production,
    capacity_factor,
    compute_power_curve,
    pitch_for_rated_power,
    weibull_cdf,
    weibull_pdf,
    weibull_scale,
)


def test_optimiser_finds_analytic_maximum():
    def bowl(rows):
        return 0.5 - (rows[:, 1] - 8.0) ** 2 - (rows[:, 2] - 1.0) ** 2 - (rows[:, 3] - 1.0) ** 2

    result = maximise_cp(bowl, 9.0, 0, seed=0, maxiter=60)
    assert result.objective_cp == pytest.approx(0.5, abs=1e-3)
    assert result.x["tsr"] == pytest.approx(8.0, abs=0.05)
    assert result.n_evaluations > 0 and result.seconds > 0
    row = raw_row(result, 9.0, 1)
    assert row.shape == (1, 6) and row[0, 0] == 9.0 and row[0, 5] == 1.0


def test_bem_optimum_beats_baseline(oracle):
    baseline = oracle.cp(np.array([[9.0, 7.5, 0.0, 1.0, 1.0, 0.0]]))[0]
    result = maximise_cp(oracle.cp, 9.0, 0, oracle.space, seed=0, popsize=6, maxiter=6)
    assert result.objective_cp >= baseline - 1e-3
    assert result.objective_cp < 16 / 27


def test_throughput_reports_positive_rates(oracle, dataset):
    rates = throughput(oracle.cp, dataset.raw[:20], repeats=1)
    assert rates["batched_per_s"] > rates["single_call_per_s"] > 0


def test_weibull_functions():
    scale = weibull_scale(10.0, 2.0)
    assert scale == pytest.approx(10.0 / 0.8862269, rel=1e-6)
    v = np.linspace(0, 80, 80001)
    pdf = weibull_pdf(v, scale)
    assert np.trapezoid(pdf, v) == pytest.approx(1.0, abs=1e-5)
    assert np.trapezoid(pdf * v, v) == pytest.approx(10.0, abs=1e-3)
    assert weibull_cdf(scale, scale) == pytest.approx(1 - np.exp(-1))


def test_aep_of_constant_power_curve():
    speed = np.array([4.0, 25.0])
    power = np.array([2.0e6, 2.0e6])
    scale = weibull_scale(8.0)
    expected = 2.0e6 * HOURS_PER_YEAR * (weibull_cdf(25.0, scale) - weibull_cdf(4.0, scale))
    aep = annual_energy_production(speed, power, 8.0)
    assert aep == pytest.approx(expected, rel=1e-4)
    assert capacity_factor(aep, 2.0e6) < 1.0
    assert annual_energy_production(speed, power, 8.0, availability=0.5) == pytest.approx(aep / 2)


def test_power_curve_with_constant_cp():
    limits = TurbineLimits()
    curve = compute_power_curve(lambda rows: np.full(len(rows), 0.45), 0, limits)
    area = np.pi * limits.radius**2
    assert curve.power[0] == pytest.approx(0.5 * 1.225 * area * 4.0**3 * 0.45 * 0.94)
    assert curve.power.max() == limits.rated_power
    assert np.all(np.diff(curve.power) >= 0)
    assert np.all(curve.tsr[np.isfinite(curve.tsr)] * curve.speed[np.isfinite(curve.tsr)] <= 90.0)
    assert 11.0 <= curve.rated_speed(limits.rated_power) <= 12.5


def test_bem_power_curve_and_rated_pitch(oracle):
    curve = compute_power_curve(oracle.cp, 0, tsr_grid=np.linspace(5, 10, 6),
                                pitch_grid=np.linspace(-1, 6, 8))
    assert 10.5 <= curve.rated_speed(10.0e6) <= 13.0
    assert 0.4 < np.nanmax(curve.cp) < 0.5
    pitch = [pitch_for_rated_power(oracle.rotors[0], u) for u in (15.0, 25.0)]
    assert 5.0 < pitch[0] < pitch[1] < 30.0
    assert np.isnan(pitch_for_rated_power(oracle.rotors[0], 6.0))
