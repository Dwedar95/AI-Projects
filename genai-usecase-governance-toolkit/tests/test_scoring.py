import pytest
from pydantic import ValidationError

from genai_governance.models import BuildBuyInputs, Portfolio
from genai_governance.scoring import (
    build_buy_partner,
    dimension_scores,
    kendall_tau,
    overall_score,
    quadrant,
    score_portfolio,
    sensitivity_analysis,
)


def test_portfolio_loads_and_is_marked_fictional(portfolio):
    assert portfolio.fictional is True
    assert len(portfolio.use_cases) == 9


def test_dimension_scores_are_weighted_and_normalised(portfolio):
    d = dimension_scores(portfolio.get("UC-01"), portfolio.scoring)
    # value criteria: (3*4 + 2*4 + 1*4) / 6 = 4 -> (4 - 1) / 4 = 0.75
    assert d["value"] == pytest.approx(0.75)
    # risk criteria: (3*2 + 2*3 + 2*3) / 7 = 18/7 -> (18/7 - 1) / 4
    assert d["risk"] == pytest.approx((18 / 7 - 1) / 4)


def test_overall_score_inverts_risk():
    w = {"value": 1.0, "feasibility": 1.0, "risk": 2.0}
    low_risk = overall_score({"value": 0.5, "feasibility": 0.5, "risk": 0.0}, w)
    high_risk = overall_score({"value": 0.5, "feasibility": 0.5, "risk": 1.0}, w)
    assert low_risk == pytest.approx(75.0)
    assert high_risk == pytest.approx(25.0)


def test_ranking_is_sorted_and_complete(portfolio):
    ranked = score_portfolio(portfolio)
    assert [r.rank for r in ranked] == list(range(1, 10))
    assert all(a.overall >= b.overall for a, b in zip(ranked, ranked[1:], strict=False))
    assert ranked[0].id == "UC-01"
    assert ranked[-1].id == "UC-06"


def test_weights_change_the_ranking(portfolio):
    value_only = score_portfolio(portfolio, {"value": 1.0, "feasibility": 0.0, "risk": 0.0})
    assert value_only[0].id == "UC-04"  # highest value dimension


def test_quadrants():
    assert quadrant(0.8, 0.8) == "quick win"
    assert quadrant(0.8, 0.2) == "strategic bet"
    assert quadrant(0.2, 0.8) == "fill-in"
    assert quadrant(0.2, 0.2) == "deprioritise"


def test_kendall_tau_extremes():
    assert kendall_tau(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert kendall_tau(["a", "b", "c"], ["c", "b", "a"]) == -1.0
    with pytest.raises(ValueError):
        kendall_tau(["a"], ["b"])


def test_sensitivity_is_reproducible_and_consistent(portfolio):
    a = sensitivity_analysis(portfolio, n_samples=300, seed=1)
    b = sensitivity_analysis(portfolio, n_samples=300, seed=1)
    assert a.mean_kendall_tau == b.mean_kendall_tau
    assert [s.mean_rank for s in a.stability] == [s.mean_rank for s in b.stability]
    assert len(a.one_at_a_time) == 6
    # every sample assigns each rank exactly once, so mean ranks sum to n(n+1)/2
    assert sum(s.mean_rank for s in a.stability) == pytest.approx(45.0)
    assert 0.0 <= a.p_top1_unchanged <= 1.0
    worst = next(s for s in a.stability if s.id == "UC-06")
    assert worst.best_rank == 9  # dominated on every dimension, always last


def test_tight_concentration_keeps_baseline_ranking(portfolio):
    rep = sensitivity_analysis(portfolio, n_samples=200, concentration=1e7)
    assert rep.mean_kendall_tau == pytest.approx(1.0)
    assert rep.p_top1_unchanged == 1.0


def test_switch_point_really_changes_the_top(portfolio):
    rep = sensitivity_analysis(portfolio, n_samples=50)
    sp = next(s for s in rep.switch_points if s["dimension"] == "value")
    assert sp["delta"] is not None and sp["new_top"] != rep.baseline[0].id


def test_build_buy_partner_recommendations():
    build = build_buy_partner(BuildBuyInputs(strategic_differentiation=5, internal_capability=5,
                                             market_solution_maturity=1, data_sensitivity=5, time_pressure=1,
                                             customisation_need=5))
    buy = build_buy_partner(BuildBuyInputs(strategic_differentiation=1, internal_capability=2,
                                           market_solution_maturity=5, data_sensitivity=1, time_pressure=5,
                                           customisation_need=1))
    partner = build_buy_partner(BuildBuyInputs(strategic_differentiation=4, internal_capability=1,
                                               market_solution_maturity=3, data_sensitivity=3, time_pressure=4,
                                               customisation_need=4))
    assert (build.recommendation, buy.recommendation, partner.recommendation) == ("build", "buy", "partner")
    assert build.margin > 0 and build.rationale


def test_neutral_inputs_are_reported_as_close_call():
    d = build_buy_partner(BuildBuyInputs(strategic_differentiation=3, internal_capability=3,
                                         market_solution_maturity=3, data_sensitivity=3, time_pressure=3,
                                         customisation_need=3))
    assert d.margin == 0
    assert any("close call" in r for r in d.rationale)


def test_validation_rejects_bad_portfolios(portfolio):
    data = portfolio.model_dump()
    data["use_cases"][0]["scores"]["financial_impact"] = 9
    with pytest.raises(ValidationError):
        Portfolio.model_validate(data)
    data = portfolio.model_dump()
    del data["use_cases"][0]["scores"]["strategic_fit"]
    with pytest.raises(ValidationError):
        Portfolio.model_validate(data)
    data = portfolio.model_dump()
    data["use_cases"][1]["depends_on"] = ["UC-99"]
    with pytest.raises(ValidationError):
        Portfolio.model_validate(data)
