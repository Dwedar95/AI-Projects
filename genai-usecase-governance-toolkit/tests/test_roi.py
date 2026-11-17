import pytest

from genai_governance.models import Economics, Range
from genai_governance.roi import (
    business_case,
    cash_flows,
    inference_cost_per_year,
    irr,
    monte_carlo,
    npv,
    payback_period,
    tornado,
)


def simple(**kw) -> Economics:
    base = dict(horizon_years=2, discount_rate=0.10, build_cost=100.0, annual_benefit_at_full_adoption=110.0,
                adoption_curve=[1.0])
    base.update(kw)
    return Economics(**base)


def test_npv_matches_hand_calculation():
    assert npv([-100, 110], 0.10) == pytest.approx(0.0)
    assert npv([-100, 60, 60], 0.0) == pytest.approx(20.0)


def test_payback_interpolates_within_year():
    assert payback_period([-100, 50, 100]) == pytest.approx(1.5)
    assert payback_period([-100, 10, 10]) is None
    assert payback_period([5, 5]) == 0.0


def test_irr_recovers_known_rate():
    assert irr([-100, 110]) == pytest.approx(0.10, abs=1e-5)
    assert irr([-100, -10]) is None


def test_inference_cost_formula():
    e = simple(requests_per_year=1000, input_tokens_per_request=2000, output_tokens_per_request=500,
               price_per_1k_input_tokens=0.01, price_per_1k_output_tokens=0.03)
    # 1000 * (2 * 0.01 + 0.5 * 0.03) = 35
    assert inference_cost_per_year(e) == pytest.approx(35.0)


def test_cash_flows_scale_with_adoption_and_hold_last_value():
    e = simple(horizon_years=3, adoption_curve=[0.5, 1.0], run_cost_per_year=10, change_management_cost=20)
    rows = cash_flows(e)
    assert rows[0].net == pytest.approx(-120.0)
    assert [r.benefit for r in rows[1:]] == pytest.approx([55.0, 110.0, 110.0])
    assert rows[1].net == pytest.approx(45.0)


def test_business_case_totals():
    bc = business_case(simple())
    assert bc.total_benefit == pytest.approx(220.0)
    assert bc.total_cost == pytest.approx(100.0)
    assert bc.roi == pytest.approx(1.2)
    assert bc.payback_years == pytest.approx(100 / 110)
    assert bc.npv == pytest.approx(-100 + 110 / 1.1 + 110 / 1.21)


def test_tornado_orders_by_swing_and_brackets_base():
    e = simple(uncertainty={"benefit": Range(low=0.5, high=1.5), "build_cost": Range(low=0.9, high=1.1)})
    bars = tornado(e)
    assert [b.driver for b in bars] == ["benefit", "build_cost"]
    base = business_case(e).npv
    assert bars[0].npv_low < base < bars[0].npv_high
    assert bars[1].npv_high < base < bars[1].npv_low  # higher build cost lowers NPV


def test_monte_carlo_without_uncertainty_equals_deterministic():
    e = simple()
    mc = monte_carlo(e, n=100)
    assert mc.npv_p10 == pytest.approx(business_case(e).npv)
    assert mc.npv_p90 == pytest.approx(business_case(e).npv)
    assert mc.prob_negative_npv == 0.0


def test_monte_carlo_is_seeded_and_ordered(portfolio):
    e = portfolio.get("UC-01").economics
    a, b, c = monte_carlo(e, n=2000, seed=3), monte_carlo(e, n=2000, seed=3), monte_carlo(e, n=2000, seed=4)
    assert a == b
    assert a.npv_p50 != c.npv_p50
    assert a.npv_p10 < a.npv_p50 < a.npv_p90
    assert 0.0 <= a.prob_negative_npv <= 1.0


def test_economics_validation():
    with pytest.raises(ValueError):
        simple(adoption_curve=[1.2])
    with pytest.raises(ValueError):
        simple(uncertainty={"weather": Range(low=0.5, high=1.5)})
    with pytest.raises(ValueError):
        Range(low=1.1, high=1.2)
