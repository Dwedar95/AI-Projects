import pytest

from process_copilot.process import Step
from process_copilot.providers import (
    LLMRequest,
    MockProvider,
    build_messages,
    count_tokens,
    parse_completion,
)
from process_copilot.retriever import Document, TfidfRetriever, load_knowledge_base, stem, tokenize
from process_copilot.routing import DEFAULT_CATALOG, BudgetExceededError, ModelRouter

from .conftest import ROOT

CHEAP, STRONG = DEFAULT_CATALOG[0], DEFAULT_CATALOG[1]
LABELS = {"refund_request": ["refund", "money back"], "complaint": ["complaint", "unhappy"]}


def test_tokenize_and_stem():
    assert stem("refunded") == "refund" and stem("refunds") == "refund" and stem("is") == "is"
    assert tokenize("The refunds were issued!") == ["refund", "were", "issu"]
    assert tokenize("The refunds", stemming=False) == ["refunds"]


def test_retriever_ranks_relevant_chunk_first():
    docs = [Document("a", "Refunds are possible within 30 days after delivery."),
            Document("b", "The payroll date is the 27th of each month."),
            Document("c", "Laptops are handed out on the first working day.")]
    r = TfidfRetriever(docs)
    hits = r.search("when is payroll paid each month", k=2)
    assert hits[0].document.doc_id == "b"
    assert 0 < hits[0].score <= 1.0
    assert r.search("quantum chromodynamics") == []
    with pytest.raises(ValueError):
        TfidfRetriever([])


def test_knowledge_base_chunks_carry_heading_and_retrieve_across_files():
    docs = load_knowledge_base(ROOT / "examples" / "knowledge")
    assert any(d.doc_id == "refund_policy#1" and d.text.startswith("Refund window:") for d in docs)
    r = TfidfRetriever(docs)
    assert r.search("How many vacation days do new employees have?", k=1)[0].document.source == "hr_handbook.md"
    assert r.search("tolerance for invoice and purchase order difference", k=1)[0].document.source == "invoice_policy.md"


def test_mock_classify_cheap_misses_inflection_strong_finds_it():
    req = LLMRequest("classify", "Classify.", "Order ORD-1 was faulty and I want it refunded.", labels=LABELS)
    cheap, strong = MockProvider().complete(CHEAP, req), MockProvider().complete(STRONG, req)
    assert cheap.data["label"] == "unknown" and cheap.confidence == 0.0
    assert strong.data["label"] == "refund_request" and strong.confidence == 1.0


def test_mock_classify_tie_gives_low_confidence_and_phrases_match():
    req = LLMRequest("classify", "Classify.", "I am unhappy and want a refund", labels=LABELS)
    assert MockProvider().complete(CHEAP, req).confidence == 0.5
    req = LLMRequest("classify", "Classify.", "I want my money back", labels=LABELS)
    assert MockProvider().complete(CHEAP, req).data["label"] == "refund_request"


def test_mock_extract_fields_and_tier_difference():
    fields = {"order_id": r"regex:ORD-\d{4}", "amount": "amount", "email": "email"}
    req = LLMRequest("extract", "Extract.", "Order ORD-1234 cost 1,250.00 USD, mail me at a.b@example.org", fields=fields)
    cheap, strong = MockProvider().complete(CHEAP, req), MockProvider().complete(STRONG, req)
    assert cheap.data == {"order_id": "ORD-1234", "amount": None, "email": "a.b@example.org"}
    assert cheap.confidence == pytest.approx(2 / 3, abs=1e-3)
    assert strong.data["amount"] == 1250.0 and strong.confidence == 1.0
    with pytest.raises(ValueError):
        MockProvider().complete(CHEAP, LLMRequest("extract", "x", "y", fields={"f": "colour"}))


def test_mock_grounded_answer_cites_source_and_reports_coverage():
    passages = [("kb#1", "Refunds are possible within 30 days after delivery. Vouchers are offered later."),
                ("kb#2", "The payroll date is the 27th.")]
    req = LLMRequest("generate", "Answer.", "How many days after delivery are refunds possible?", passages=passages)
    resp = MockProvider().complete(CHEAP, req)
    assert "30 days" in resp.text and resp.data["sources"] == ["kb#1"]
    assert resp.confidence > 0.6
    none = MockProvider().complete(CHEAP, LLMRequest("generate", "Answer.", "zebra crossing", passages=passages))
    assert none.confidence == 0.0 and none.data["sources"] == []


def test_mock_is_deterministic_and_accounts_tokens():
    req = LLMRequest("summarise", "Summarise.", "First sentence here. Second one. Third one.")
    a, b = MockProvider().complete(STRONG, req), MockProvider().complete(STRONG, req)
    assert a == b
    assert a.text == "First sentence here. Second one."
    assert a.input_tokens == count_tokens("Summarise.") + count_tokens(req.text)
    assert a.latency_ms == STRONG.base_latency_ms + STRONG.ms_per_output_token * a.output_tokens
    assert MockProvider().complete(CHEAP, req).text == "First sentence here."


def test_real_adapter_message_building_and_parsing_offline():
    req = LLMRequest("classify", "Classify.", "text", labels=LABELS)
    messages = build_messages(req)
    assert messages[0]["role"] == "system" and "refund_request" in messages[0]["content"]
    payload = {"choices": [{"message": {"content": '{"label": "complaint", "confidence": 0.8}'}}],
               "usage": {"prompt_tokens": 50, "completion_tokens": 9}}
    resp = parse_completion(payload, req, STRONG, 12.0)
    assert resp.data == {"label": "complaint"} and resp.confidence == 0.8
    assert (resp.input_tokens, resp.output_tokens) == (50, 9)
    bad = parse_completion({"choices": [{"message": {"content": "not json"}}]}, req, STRONG, 1.0)
    assert bad.confidence == 0.0


def _step(**kw):
    base = {"id": "s", "type": "prompt", "task": "classify", "prompt": "p", "labels": {"a": [], "b": []}}
    base.update(kw)
    return Step(**base)


def test_router_tier_by_complexity():
    router = ModelRouter()
    assert router.route(_step(), 50, 0.0, 1.0).model.name == "prop-small"
    assert router.route(_step(task="generate", labels={}), 50, 0.0, 1.0).model.name == "prop-large"
    assert router.route(_step(task="summarise", labels={}), 500, 0.0, 1.0).model.tier == "strong"  # length bonus
    assert router.route(_step(complexity="high"), 50, 0.0, 1.0).model.tier == "strong"
    assert router.route(_step(task="generate", labels={}, complexity="low"), 50, 0.0, 1.0).model.tier == "cheap"


def test_router_sends_confidential_data_to_open_source_models():
    d = ModelRouter().route(_step(data_classification="confidential"), 50, 0.0, 1.0)
    assert d.model.hosting == "open_source" and "self-hosted" in d.reason


def test_router_budget_downgrade_and_exhaustion():
    router = ModelRouter()
    step = _step(task="generate", labels={})
    strong_cost = STRONG.cost(200, step.max_output_tokens)
    d = router.route(step, 200, spent=0.0, budget=strong_cost * 0.5)
    assert d.model.tier == "cheap" and d.downgraded and "downgraded" in d.reason
    with pytest.raises(BudgetExceededError):
        router.route(step, 200, spent=1.0, budget=1.0)


def test_router_escalation_respects_tier_and_budget():
    router = ModelRouter()
    step = _step()
    cheap = router.route(step, 50, 0.0, 1.0)
    up = router.escalate(cheap, step, 50, 0.0, 1.0)
    assert up.model.name == "prop-large"
    assert router.escalate(up, step, 50, 0.0, 1.0) is None
    assert router.escalate(cheap, step, 50, spent=1.0, budget=1.0) is None
    with pytest.raises(ValueError):
        ModelRouter(catalog=DEFAULT_CATALOG[:3])
