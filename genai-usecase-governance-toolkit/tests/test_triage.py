import pytest

from genai_governance.risk_triage import DISCLAIMER, QUESTIONS, infer_answers, triage


def refs(result):
    return {o.reference for o in result.obligations}


def test_prohibited_takes_precedence():
    r = triage("anything", {"social_scoring": True, "employment": True, "interacts_with_people": True})
    assert r.tier == "prohibited"
    assert refs(r) == {"Art. 5"}
    assert any("Art. 5(1)(c)" in reason for reason in r.reasons)


def test_keyword_inference_detects_recruitment():
    assert infer_answers("We use CV screening to rank candidates")["employment"] is True
    r = triage("We use CV screening to rank candidates")
    assert r.tier == "high-risk"
    assert "employment" in r.inferred_from_description
    assert "keyword match" in r.reasons[0]


def test_explicit_answer_overrides_keyword():
    r = triage("Recruiting newsletter text generation", {"employment": False})
    assert r.tier == "limited"
    assert "employment" not in r.inferred_from_description


def test_deployer_and_provider_get_different_checklists():
    dep = triage("", {"essential_services": True}, role="deployer")
    prov = triage("", {"essential_services": True}, role="provider")
    assert {"Art. 26(1)", "Art. 26(6)", "Art. 27", "Art. 26(11)", "Art. 4"} <= refs(dep)
    assert {"Art. 9", "Art. 10", "Art. 14", "Art. 43", "Art. 49"} <= refs(prov)
    assert "Art. 9" not in refs(dep)


def test_workplace_notification_only_when_used_at_workplace():
    assert "Art. 26(7)" not in refs(triage("", {"employment": True}))
    assert "Art. 26(7)" in refs(triage("", {"employment": True, "used_in_workplace": True}))


def test_derogation_applies_without_profiling():
    r = triage("", {"employment": True, "narrow_procedural_task": True})
    assert r.tier == "minimal" and r.derogation_applied
    assert {"Art. 6(4)", "Art. 49(2)"} <= refs(r)


def test_derogation_blocked_by_profiling():
    r = triage("", {"employment": True, "narrow_procedural_task": True, "profiling_of_persons": True})
    assert r.tier == "high-risk" and not r.derogation_applied
    assert any("profiling" in reason for reason in r.reasons)


def test_derogation_does_not_apply_to_annex_i_products():
    r = triage("", {"safety_component_regulated_product": True, "preparatory_task": True})
    assert r.tier == "high-risk"


def test_limited_and_minimal_tiers():
    chat = triage("A chatbot for customers")
    assert chat.tier == "limited" and "Art. 50(1)" in refs(chat)
    plain = triage("Forecast spare part demand from order history")
    assert plain.tier == "minimal"
    assert refs(plain) == {"Art. 4"}


def test_high_risk_chatbot_keeps_transparency_obligation():
    r = triage("A chatbot", {"education": True})
    assert r.tier == "high-risk" and "Art. 50(1)" in refs(r)


def test_input_validation_and_disclaimer():
    with pytest.raises(ValueError):
        triage("", {"not_a_question": True})
    with pytest.raises(ValueError):
        triage("", {}, role="importer")
    assert "not legal advice" in DISCLAIMER
    assert triage("x").disclaimer == DISCLAIMER


def test_question_ids_unique_and_referenced():
    ids = [q.id for q in QUESTIONS]
    assert len(ids) == len(set(ids))
    assert all(q.reference.startswith("Art.") for q in QUESTIONS)


def test_example_portfolio_tiers(portfolio):
    tiers = {u.id: triage(u.description, u.triage, u.role).tier for u in portfolio.use_cases}
    assert tiers["UC-03"] == "high-risk"
    assert tiers["UC-06"] == "prohibited"
    assert tiers["UC-05"] == "minimal"
    assert tiers["UC-09"] == "limited"  # Annex III area, derogation, but still a chatbot
