"""The RAG pipeline: ingest, retrieve, guard, generate."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from rag_eval_kit.chunking import get_chunker
from rag_eval_kit.documents import Chunk, Document
from rag_eval_kit.embeddings import Embedder, get_embedder
from rag_eval_kit.generation import Answer, AnswerGenerator, AnswerSentence, ExtractiveGenerator
from rag_eval_kit.retrieval import HybridRetriever, Mode, SearchHit
from rag_eval_kit.safety import detect_injection, redact_pii

BLOCKED_TEXT = "This request was blocked because it looks like an attempt to override the assistant's instructions."


@dataclass(frozen=True)
class PipelineConfig:
    """Pipeline settings. The defaults are the configuration evaluated in the README."""

    chunker: str = "heading"
    embedder: str = "hashing"
    mode: Mode = "hybrid"
    rerank: bool = True
    top_k: int = 5
    block_injection: bool = True
    quarantine_injected_chunks: bool = True
    redact_output_pii: bool = True


@dataclass(frozen=True)
class RagResponse:
    """Result of one question, with the evidence and guardrail decisions that led to it."""

    question: str
    answer: Answer
    hits: list[SearchHit]
    blocked: bool = False
    flags: tuple[str, ...] = ()
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def latency_ms(self) -> float:
        return sum(self.timings_ms.values())


class RagPipeline:
    """Wires chunking, hybrid retrieval, guardrails and answer generation together."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        embedder: Embedder | None = None,
        generator: AnswerGenerator | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.retriever = HybridRetriever(embedder or get_embedder(self.config.embedder))
        self._generator = generator
        self.documents: dict[str, Document] = {}
        self.chunks: list[Chunk] = []
        self.quarantined: set[str] = set()

    @property
    def generator(self) -> AnswerGenerator:
        if self._generator is None:
            self._generator = ExtractiveGenerator(self.retriever.bm25.idf)
        return self._generator

    def ingest(self, documents: list[Document]) -> int:
        """Add or replace documents, rebuild the indexes and return the chunk count."""
        for document in documents:
            self.documents[document.doc_id] = document
        chunker = get_chunker(self.config.chunker)
        self.chunks = [chunk for document in self.documents.values() for chunk in chunker.chunk(document)]
        self.retriever.index(self.chunks)
        self.quarantined = {c.chunk_id for c in self.chunks if detect_injection(c.text).flagged}
        if isinstance(self._generator, ExtractiveGenerator):
            self._generator.idf = self.retriever.bm25.idf
        return len(self.chunks)

    def search(
        self, query: str, k: int | None = None, mode: Mode | None = None, rerank: bool | None = None
    ) -> list[SearchHit]:
        """Retrieve chunks with the configured or the given mode."""
        if not self.chunks:
            return []
        return self.retriever.search(
            query,
            k or self.config.top_k,
            mode or self.config.mode,
            self.config.rerank if rerank is None else rerank,
        )

    def answer(
        self, question: str, k: int | None = None, mode: Mode | None = None, rerank: bool | None = None
    ) -> RagResponse:
        """Answer a question with guardrails applied before and after generation."""
        flags: list[str] = []
        started = time.perf_counter()
        injection = detect_injection(question)
        if injection.flagged:
            flags.append("prompt_injection:" + ",".join(injection.matched))
            if self.config.block_injection:
                blocked = Answer(text=BLOCKED_TEXT, refused=True)
                timings = {"guard": (time.perf_counter() - started) * 1000}
                return RagResponse(question, blocked, [], True, tuple(flags), timings)
        guard_done = time.perf_counter()

        hits = self.search(question, k, mode, rerank)
        retrieved = time.perf_counter()

        context = hits
        if self.config.quarantine_injected_chunks:
            context = [hit for hit in hits if hit.chunk.chunk_id not in self.quarantined]
            flags += [f"quarantined_chunk:{hit.chunk.chunk_id}" for hit in hits if hit not in context]
        answer = self.generator.generate(question, context)
        if self.config.redact_output_pii and not answer.refused:
            redacted = redact_pii(answer.text)
            if redacted != answer.text:
                flags.append("output_pii_redacted")
                sentences = tuple(AnswerSentence(redact_pii(s.text), s.markers) for s in answer.sentences)
                answer = Answer(redacted, sentences, answer.citations, answer.refused, answer.metadata)
        done = time.perf_counter()
        timings = {
            "guard": (guard_done - started) * 1000,
            "retrieve": (retrieved - guard_done) * 1000,
            "generate": (done - retrieved) * 1000,
        }
        return RagResponse(question, answer, hits, False, tuple(flags), timings)
