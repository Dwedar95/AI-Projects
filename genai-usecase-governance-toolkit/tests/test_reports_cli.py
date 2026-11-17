import pytest

from genai_governance.cards import system_card
from genai_governance.cli import main
from genai_governance.fairness import fairness_report, load_predictions, performance_metrics
from genai_governance.models import Portfolio
from genai_governance.reports import (
    ROLES,
    build_raci,
    build_reports,
    build_risk_register,
    plan_waves,
    triage_portfolio,
)
from genai_governance.risk_triage import triage


def test_waves_respect_capacity_exclusion_and_dependencies(portfolio):
    planned, excluded = plan_waves(portfolio, capacity=2, n_waves=3)
    waves = {a.use_case_id: a.wave for a in planned}
    assert [e.use_case_id for e in excluded] == ["UC-06"]
    assert len(waves) == 8
    for w in (1, 2):
        assert sum(1 for v in waves.values() if v == w) <= 2
    assert waves["UC-01"] == 1
    assert waves["UC-04"] >= 2  # strategic bet
    assert waves["UC-03"] >= 2  # high-risk
    assert waves["UC-08"] >= waves["UC-01"]  # dependency


def test_dependency_pulls_use_case_to_later_wave(portfolio):
    data = portfolio.model_dump()
    for uc in data["use_cases"]:
        if uc["id"] == "UC-05":
            uc["depends_on"] = ["UC-03"]  # feasible fill-in now waits for a high-risk item
    p = Portfolio.model_validate(data)
    waves = {a.use_case_id: a for a in plan_waves(p, capacity=5)[0]}
    assert waves["UC-05"].wave >= waves["UC-03"].wave
    assert "dependency UC-03" in waves["UC-05"].reason


def test_dependency_cycle_and_dependency_on_excluded(portfolio):
    data = portfolio.model_dump()
    by = {u["id"]: u for u in data["use_cases"]}
    by["UC-01"]["depends_on"], by["UC-08"]["depends_on"] = ["UC-08"], ["UC-01"]
    with pytest.raises(ValueError, match="cycle"):
        plan_waves(Portfolio.model_validate(data))
    data = portfolio.model_dump()
    by = {u["id"]: u for u in data["use_cases"]}
    by["UC-02"]["depends_on"] = ["UC-06"]
    _, excluded = plan_waves(Portfolio.model_validate(data))
    assert {e.use_case_id for e in excluded} == {"UC-06", "UC-02"}


def test_risk_register_is_sorted_and_rated(portfolio):
    entries = build_risk_register(portfolio)
    assert [e.severity for e in entries] == sorted((e.severity for e in entries), reverse=True)
    assert entries[0].severity == 25 and entries[0].rating == "high"  # prohibited practice
    hr = [e for e in entries if e.use_case_id == "UC-03" and "high-risk" in e.risk]
    assert hr and hr[0].owner == "Legal and compliance"
    assert all(1 <= e.likelihood <= 5 and 1 <= e.impact <= 5 for e in entries)


def test_fairness_flags_enter_the_risk_register(portfolio, predictions_path):
    rep = fairness_report(load_predictions(predictions_path))
    entries = build_risk_register(portfolio, fairness={"UC-03": rep})
    assert any(e.use_case_id == "UC-03" and e.risk.startswith("Fairness thresholds") for e in entries)


def test_raci_has_one_accountable_per_activity():
    for tier in ("minimal", "limited", "high-risk"):
        rows = build_raci(tier)
        for _activity, mapping in rows:
            assert list(mapping) == list(ROLES)
            assert list(mapping.values()).count("A") == 1
            assert "R" in mapping.values()
    assert len(build_raci("high-risk")) > len(build_raci("limited")) > len(build_raci("minimal"))


def test_build_reports_content(portfolio):
    reports = build_reports(portfolio)
    assert set(reports) == {"roadmap.md", "risk_register.md", "raci.md"}
    assert "## Wave 1" in reports["roadmap.md"] and "fictional" in reports["roadmap.md"]
    assert "not legal advice" in reports["risk_register.md"]
    assert "## Tier: high-risk" in reports["raci.md"]
    for text in reports.values():
        assert chr(8212) not in text and chr(8211) not in text  # no em or en dashes


def test_system_card_includes_computed_metrics(portfolio, predictions_path):
    uc = portfolio.get("UC-03")
    data = load_predictions(predictions_path)
    perf, fair = performance_metrics(data), fairness_report(data)
    card = system_card(uc, triage(uc.description, uc.triage, uc.role), perf, fair, evaluation_data_note="synthetic")
    assert f"| Accuracy | {perf.accuracy:.3f} |" in card
    assert "Indicative tier: **high-risk**" in card
    assert "Disparate impact ratio" in card and "synthetic" in card
    bare = system_card(uc, triage_portfolio(portfolio)["UC-03"])
    assert "No evaluation data was supplied" in bare


def test_cli_prioritise_and_roi(portfolio_path, capsys):
    assert main(["prioritise", str(portfolio_path)]) == 0
    out = capsys.readouterr().out
    assert "fictional example data" in out and "UC-01" in out
    assert main(["roi", str(portfolio_path), "--use-case", "UC-01", "--samples", "200"]) == 0
    assert "NPV P10 / P50 / P90" in capsys.readouterr().out
    assert main(["roi", str(portfolio_path), "--use-case", "UC-02"]) == 2
    assert main(["roi", str(portfolio_path), "--use-case", "NOPE"]) == 2


def test_cli_triage_fairness_report_card(portfolio_path, predictions_path, tmp_path, capsys):
    assert main(["triage", "--describe", "police tool", "--yes", "profiling_of_persons"]) == 0
    out = capsys.readouterr().out
    assert "HIGH-RISK" in out and "not legal advice" in out
    assert main(["triage", "--describe", "x", "--yes", "bogus"]) == 2
    assert main(["fairness", str(predictions_path), "--bootstrap", "50", "--fail-on-flags"]) == 1
    assert main(["sensitivity", str(portfolio_path), "--samples", "100"]) == 0
    assert main(["build-buy", str(portfolio_path), "--use-case", "UC-04"]) == 0
    capsys.readouterr()
    assert main(["report", str(portfolio_path), "--out", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "roadmap.md").exists()
    card = tmp_path / "card.md"
    assert main(["card", str(portfolio_path), "--use-case", "UC-03", "--predictions", str(predictions_path),
                 "--bootstrap", "50", "--out", str(card)]) == 0
    assert card.read_text(encoding="utf-8").startswith("# System card")
