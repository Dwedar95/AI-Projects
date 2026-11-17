"""Runs the evaluation: retrieval comparison, answer quality, safety probes and latency."""

from __future__ import annotations

import json
import platform
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from rag_eval_kit import __version__
from rag_eval_kit.documents import Document, bundled_corpus_dir, load_corpus
from rag_eval_kit.evaluation import metrics as m
from rag_eval_kit.pipeline import PipelineConfig, RagPipeline
from rag_eval_kit.retrieval import Mode
from rag_eval_kit.safety import detect_injection, find_pii

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RETRIEVAL_DEPTH = 10
OFFLINE_EMBEDDERS = ("hashing", "tfidf-svd")
RETRIEVAL_CONFIGS: tuple[tuple[str, Mode, bool], ...] = (
    ("bm25", "bm25", False),
    ("dense", "dense", False),
    ("hybrid", "hybrid", False),
    ("hybrid+rerank", "hybrid", True),
)


@dataclass(frozen=True)
class QAExample:
    id: str
    question: str
    relevant_docs: tuple[str, ...]
    answer_keywords: tuple[str, ...]
    answerable: bool
    kind: str


@dataclass
class EvalReport:
    """All measured values plus the context needed to reproduce them."""

    meta: dict[str, Any]
    retrieval: dict[str, dict[str, float]]
    retrieval_by_kind: dict[str, dict[str, float]]
    embedders: dict[str, dict[str, float]]
    chunking: dict[str, dict[str, float]]
    answers: dict[str, float]
    safety: dict[str, float]
    latency: dict[str, float]
    failures: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": self.meta,
            "retrieval": self.retrieval,
            "retrieval_by_kind": self.retrieval_by_kind,
            "embedders": self.embedders,
            "chunking": self.chunking,
            "answers": self.answers,
            "safety": self.safety,
            "latency": self.latency,
            "failures": self.failures,
        }

    def metric(self, path: str) -> float | None:
        """Look up a metric by dotted path, for example ``retrieval.hybrid.recall@5``."""
        current: Any = self.to_dict()
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]
        return float(current) if isinstance(current, int | float) else None


def load_qa(path: str | Path | None = None) -> list[QAExample]:
    """Load a QA set from JSON Lines. Without a path the bundled synthetic set is used."""
    source = Path(path) if path else DATA_DIR / "qa_synthetic.jsonl"
    examples = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            examples.append(
                QAExample(
                    id=row["id"],
                    question=row["question"],
                    relevant_docs=tuple(row.get("relevant_docs", [])),
                    answer_keywords=tuple(row.get("answer_keywords", [])),
                    answerable=bool(row.get("answerable", True)),
                    kind=row.get("kind", "general"),
                )
            )
    return examples


def load_probes(path: str | Path | None = None) -> dict[str, Any]:
    """Load the safety probes. Without a path the bundled synthetic probes are used."""
    return json.loads((Path(path) if path else DATA_DIR / "safety_probes.json").read_text(encoding="utf-8"))


def retrieval_metrics(pipeline: RagPipeline, examples: list[QAExample], mode: Mode, rerank: bool) -> dict[str, float]:
    """Document-level retrieval metrics averaged over the answerable examples."""
    rows: dict[str, list[float]] = {name: [] for name in ("recall@1", "recall@3", "recall@5", "mrr@10", "ndcg@10")}
    for example in examples:
        if not example.answerable:
            continue
        hits = pipeline.search(example.question, RETRIEVAL_DEPTH, mode, rerank)
        ranked = m.dedupe(hit.chunk.doc_id for hit in hits)
        relevant = set(example.relevant_docs)
        for k in (1, 3, 5):
            rows[f"recall@{k}"].append(m.recall_at_k(ranked, relevant, k))
        rows["mrr@10"].append(m.reciprocal_rank(ranked, relevant))
        rows["ndcg@10"].append(m.ndcg_at_k(ranked, relevant, 10))
    return {name: round(m.mean(values), 4) for name, values in rows.items()} | {"n": len(rows["mrr@10"])}


def answer_metrics(pipeline: RagPipeline, examples: list[QAExample]) -> tuple[dict[str, float], list[dict[str, str]]]:
    """End-to-end answer quality with the pipeline's configured retrieval and generator."""
    grounded, precision, recall = [], [], []
    answered = Counter()
    correct = Counter()
    totals = Counter()
    refused_unanswerable = 0
    failures: list[dict[str, str]] = []
    for example in examples:
        response = pipeline.answer(example.question)
        answer = response.answer
        totals[example.kind] += 1
        if not example.answerable:
            refused_unanswerable += answer.refused
            if not answer.refused:
                failures.append({"id": example.id, "kind": example.kind, "issue": "answered an unanswerable question"})
            continue
        if answer.refused:
            failures.append({"id": example.id, "kind": example.kind, "issue": "refused an answerable question"})
            continue
        answered[example.kind] += 1
        relevant = set(example.relevant_docs)
        grounded.append(m.groundedness(answer, response.hits))
        cited = m.citation_precision(answer, relevant)
        if cited is not None:
            precision.append(cited)
        recall.append(m.citation_recall(answer, relevant))
        if m.contains_keywords(answer.text, example.answer_keywords):
            correct[example.kind] += 1
        else:
            failures.append({"id": example.id, "kind": example.kind, "issue": "answer misses the expected key phrase"})
    answerable_kinds = sorted({e.kind for e in examples if e.answerable})
    n_answerable = sum(totals[k] for k in answerable_kinds)
    n_unanswerable = sum(1 for e in examples if not e.answerable)
    n_answered = sum(answered.values())
    result = {
        "n_answerable": n_answerable,
        "n_unanswerable": n_unanswerable,
        "answer_rate": round(n_answered / n_answerable, 4) if n_answerable else 0.0,
        "groundedness": round(m.mean(grounded), 4),
        "citation_precision": round(m.mean(precision), 4),
        "citation_recall": round(m.mean(recall), 4),
        "keyword_accuracy_of_answered": round(sum(correct.values()) / n_answered, 4) if n_answered else 0.0,
        "keyword_accuracy_end_to_end": round(sum(correct.values()) / n_answerable, 4) if n_answerable else 0.0,
        "unanswerable_refusal_rate": round(refused_unanswerable / n_unanswerable, 4) if n_unanswerable else 1.0,
    }
    for kind in answerable_kinds:
        result[f"answer_rate_{kind}"] = round(answered[kind] / totals[kind], 4)
        result[f"keyword_accuracy_end_to_end_{kind}"] = round(correct[kind] / totals[kind], 4)
    return result, failures


def safety_metrics(pipeline: RagPipeline, probes: dict[str, Any], benign_questions: list[str]) -> dict[str, float]:
    """Prompt injection and PII metrics on labelled probes and on the live pipeline."""
    attacks = probes["injection_attacks"]
    benign = list(probes["injection_benign"]) + benign_questions
    detected = sum(detect_injection(text).flagged for text in attacks)
    false_alarms = sum(detect_injection(text).flagged for text in benign)
    blocked = sum(pipeline.answer(text).blocked for text in attacks)

    tp = fp = fn = 0
    for sample in probes["pii_positive"]:
        expected, found = Counter(sample["kinds"]), Counter(match.kind for match in find_pii(sample["text"]))
        tp += sum((expected & found).values())
        fn += sum((expected - found).values())
        fp += sum((found - expected).values())
    fp += sum(len(find_pii(text)) for text in probes["pii_negative"])
    pii_precision, pii_recall = m.precision_recall(tp, fp, fn)

    corpus_pii = {match.value for doc in pipeline.documents.values() for match in find_pii(doc.text)}
    leaks = sum(any(v in pipeline.answer(q).answer.text for v in corpus_pii) for q in probes["pii_leak_queries"])

    canary = probes["canary"]
    unguarded = RagPipeline(replace(pipeline.config, quarantine_injected_chunks=False))
    unguarded.ingest(list(pipeline.documents.values()))
    queries = probes["indirect_injection_queries"]
    return {
        "injection_attacks": len(attacks),
        "injection_benign": len(benign),
        "injection_recall": round(detected / len(attacks), 4),
        "injection_false_positive_rate": round(false_alarms / len(benign), 4),
        "injection_block_rate": round(blocked / len(attacks), 4),
        "pii_entities": tp + fn,
        "pii_precision": round(pii_precision, 4),
        "pii_recall": round(pii_recall, 4),
        "pii_output_leaks": leaks,
        "pii_leak_queries": len(probes["pii_leak_queries"]),
        "indirect_injection_queries": len(queries),
        "indirect_injection_leaks": sum(canary in pipeline.answer(q).answer.text for q in queries),
        "indirect_injection_leaks_without_quarantine": sum(canary in unguarded.answer(q).answer.text for q in queries),
        "quarantined_chunks": len(pipeline.quarantined),
    }


def latency_metrics(pipeline: RagPipeline, examples: list[QAExample], repeats: int = 5) -> dict[str, float]:
    """Wall-clock latency of ``pipeline.answer`` in milliseconds. Depends on the machine."""
    for example in examples[:5]:
        pipeline.answer(example.question)
    samples = [pipeline.answer(e.question).latency_ms for _ in range(repeats) for e in examples]
    return {
        "samples": len(samples),
        "mean_ms": round(m.mean(samples), 3),
        "p50_ms": round(m.percentile(samples, 50), 3),
        "p95_ms": round(m.percentile(samples, 95), 3),
        "p99_ms": round(m.percentile(samples, 99), 3),
    }


def run_evaluation(
    documents: list[Document] | None = None,
    examples: list[QAExample] | None = None,
    probes: dict[str, Any] | None = None,
    config: PipelineConfig | None = None,
    latency_repeats: int = 5,
) -> EvalReport:
    """Evaluate a pipeline configuration. Defaults: bundled synthetic corpus, QA set and probes."""
    bundled = documents is None and examples is None
    documents = documents if documents is not None else load_corpus(bundled_corpus_dir())
    examples = examples if examples is not None else load_qa()
    probes = probes if probes is not None else load_probes()
    config = config or PipelineConfig()

    pipeline = RagPipeline(config)
    chunk_count = pipeline.ingest(documents)

    retrieval = {name: retrieval_metrics(pipeline, examples, mode, rerank) for name, mode, rerank in RETRIEVAL_CONFIGS}
    by_kind: dict[str, dict[str, float]] = {}
    for kind in sorted({e.kind for e in examples if e.answerable}):
        subset = [e for e in examples if e.kind == kind]
        by_kind[kind] = {
            name: retrieval_metrics(pipeline, subset, mode, rerank)["recall@5"]
            for name, mode, rerank in RETRIEVAL_CONFIGS
        } | {"n": len(subset)}

    embedders: dict[str, dict[str, float]] = {}
    for name in OFFLINE_EMBEDDERS:
        variant = pipeline if name == config.embedder else RagPipeline(replace(config, embedder=name))
        if variant is not pipeline:
            variant.ingest(documents)
        embedders[name] = {
            f"{label} {metric}": retrieval_metrics(variant, examples, mode, rerank)[metric]
            for label, mode, rerank in RETRIEVAL_CONFIGS[1:]
            for metric in ("recall@5", "mrr@10")
        }

    chunking: dict[str, dict[str, float]] = {}
    for strategy in ("fixed", "sentence", "heading"):
        variant = pipeline if strategy == config.chunker else RagPipeline(replace(config, chunker=strategy))
        chunks = chunk_count if variant is pipeline else variant.ingest(documents)
        scores = retrieval_metrics(variant, examples, config.mode, config.rerank)
        chunking[strategy] = {"chunks": chunks, "recall@5": scores["recall@5"], "mrr@10": scores["mrr@10"]}

    answers, failures = answer_metrics(pipeline, examples)
    meta = {
        "version": __version__,
        "python": platform.python_version(),
        "documents": len(documents),
        "chunks": chunk_count,
        "qa_examples": len(examples),
        "qa_by_kind": dict(sorted(Counter(e.kind for e in examples).items())),
        "config": {
            "chunker": config.chunker,
            "embedder": config.embedder,
            "mode": config.mode,
            "rerank": config.rerank,
            "top_k": config.top_k,
        },
        "generator": type(pipeline.generator).__name__,
        "data": "bundled synthetic corpus and QA set" if bundled else "custom corpus or QA set",
    }
    return EvalReport(
        meta=meta,
        retrieval=retrieval,
        retrieval_by_kind=by_kind,
        embedders=embedders,
        chunking=chunking,
        answers=answers,
        safety=safety_metrics(pipeline, probes, [e.question for e in examples]),
        latency=latency_metrics(pipeline, examples, latency_repeats),
        failures=failures,
    )
