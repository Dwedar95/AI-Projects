import math

import pytest

from rag_eval_kit.documents import Chunk
from rag_eval_kit.evaluation import apply_gate, load_probes, load_qa, load_thresholds, render_markdown
from rag_eval_kit.evaluation import metrics as m
from rag_eval_kit.generation import Answer, AnswerSentence, Citation
from rag_eval_kit.retrieval import SearchHit


def test_recall_mrr_ndcg_hand_values():
    ranked, relevant = ["a", "b", "c", "d"], {"b", "d"}
    assert m.recall_at_k(ranked, relevant, 1) == 0.0
    assert m.recall_at_k(ranked, relevant, 2) == 0.5
    assert m.recall_at_k(ranked, relevant, 4) == 1.0
    assert m.reciprocal_rank(ranked, relevant) == 0.5
    assert m.reciprocal_rank(ranked, {"z"}) == 0.0
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    ideal = 1 / math.log2(2) + 1 / math.log2(3)
    assert m.ndcg_at_k(ranked, relevant, 4) == pytest.approx(dcg / ideal)
    assert m.ndcg_at_k(["b", "d"], relevant, 2) == pytest.approx(1.0)
    assert m.dedupe(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]
    with pytest.raises(ValueError):
        m.recall_at_k(ranked, set(), 3)


def test_percentile_interpolates():
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert m.percentile(values, 50) == 30.0
    assert m.percentile(values, 95) == pytest.approx(48.0)
    assert m.percentile(values, 0) == 10.0 and m.percentile(values, 100) == 50.0
    assert m.percentile([7.0], 99) == 7.0
    with pytest.raises(ValueError):
        m.percentile([], 50)


def _hit(chunk_id, text):
    return SearchHit(Chunk(chunk_id, chunk_id.split("::")[0], "T", "S", text, 0), 1.0)


def test_groundedness_detects_unsupported_and_uncited_sentences():
    hits = [_hit("a::0", "Daily backups are kept for 35 days."), _hit("b::0", "Invoices are retained for 10 years.")]
    citations = (Citation(1, "a::0", "a", "T", "S"), Citation(2, "b::0", "b", "T", "S"))
    grounded = Answer(
        "x",
        (
            AnswerSentence("Backups are kept for 35 days.", (1,)),
            AnswerSentence("Invoices are retained 10 years.", (2,)),
        ),
        citations,
    )
    assert m.groundedness(grounded, hits) == 1.0
    hallucinated = Answer(
        "x",
        (
            AnswerSentence("Backups are kept for 35 days.", (1,)),
            AnswerSentence("Backups are also stored on tape in Zurich.", (1,)),
            AnswerSentence("Invoices are retained for 10 years.", (1,)),  # true, but cites the wrong chunk
            AnswerSentence("Invoices are retained for 10 years.", ()),  # no citation
        ),
        citations,
    )
    assert m.groundedness(hallucinated, hits) == 0.25
    assert m.groundedness(Answer("refused", refused=True), hits) == 0.0


def test_citation_metrics_and_keywords():
    answer = Answer("x", (), (Citation(1, "a::0", "a", "T", "S"), Citation(2, "c::0", "c", "T", "S")))
    assert m.citation_precision(answer, {"a", "b"}) == 0.5
    assert m.citation_recall(answer, {"a", "b"}) == 0.5
    assert m.citation_precision(Answer("x"), {"a"}) is None
    assert m.contains_keywords("The limit is 150 Euros per night", ["150 euros", "night"])
    assert not m.contains_keywords("The limit is 150 euros", ["150 euros", "breakfast"])
    assert m.precision_recall(8, 2, 0) == (0.8, 1.0)
    assert m.precision_recall(0, 0, 0) == (1.0, 1.0)


def test_bundled_data_sets_are_consistent(documents):
    examples = load_qa()
    doc_ids = {d.doc_id for d in documents}
    assert len(examples) == 58 and len({e.id for e in examples}) == 58
    for example in examples:
        assert set(example.relevant_docs) <= doc_ids
        assert bool(example.relevant_docs) == example.answerable
    probes = load_probes()
    assert "Synthetic" in probes["_note"]
    assert any(probes["canary"] in d.text for d in documents)


def test_report_structure_and_value_ranges(report):
    assert set(report.retrieval) == {"bm25", "dense", "hybrid", "hybrid+rerank"}
    for scores in report.retrieval.values():
        assert scores["n"] == 52
        assert 0 <= scores["recall@1"] <= scores["recall@3"] <= scores["recall@5"] <= 1
        assert 0 <= scores["mrr@10"] <= 1 and 0 <= scores["ndcg@10"] <= 1
    assert set(report.chunking) == {"fixed", "sentence", "heading"}
    assert report.meta["chunks"] == report.chunking["heading"]["chunks"]
    assert report.latency["p50_ms"] <= report.latency["p95_ms"] <= report.latency["p99_ms"]
    assert report.metric("retrieval.hybrid+rerank.recall@5") == report.retrieval["hybrid+rerank"]["recall@5"]
    assert report.metric("retrieval.nope.recall@5") is None


def test_keyword_questions_are_retrieved_and_safety_holds(report):
    assert report.retrieval_by_kind["keyword"]["bm25"] == 1.0
    assert report.answers["groundedness"] == 1.0
    assert report.answers["unanswerable_refusal_rate"] == 1.0
    assert report.safety["pii_output_leaks"] == 0
    assert report.safety["indirect_injection_leaks"] == 0
    assert report.safety["indirect_injection_leaks_without_quarantine"] >= 1


def test_gate_passes_bundled_thresholds_and_fails_on_strict_or_missing(report):
    gate = apply_gate(report)
    assert gate.passed and len(gate.checks) == 11
    strict = apply_gate(report, [{"metric": "retrieval.bm25.recall@1", "op": ">=", "threshold": 0.999}])
    assert not strict.passed and strict.failed[0].actual == report.retrieval["bm25"]["recall@1"]
    missing = apply_gate(report, [{"metric": "answers.does_not_exist", "op": ">=", "threshold": 0}])
    assert not missing.passed and missing.to_dict()["checks"][0]["actual"] is None


def test_threshold_file_validation_and_markdown(report, tmp_path):
    bad = tmp_path / "gate.yaml"
    bad.write_text("checks:\n  - {metric: a.b, op: '~=', threshold: 1}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown gate operator"):
        load_thresholds(bad)
    markdown = render_markdown(report, apply_gate(report))
    assert "## Retrieval: BM25 vs dense vs hybrid" in markdown
    assert "| hybrid+rerank |" in markdown and "Gate: PASS" in markdown
    assert "synthetic" in markdown
