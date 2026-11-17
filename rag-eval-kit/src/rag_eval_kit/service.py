"""FastAPI service: API-key authentication, role checks and an audit trail."""

from __future__ import annotations

import hashlib
import hmac
import os
import uuid
from dataclasses import dataclass
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from pydantic import BaseModel, Field

from rag_eval_kit.audit import AuditLog
from rag_eval_kit.documents import Document
from rag_eval_kit.pipeline import RagPipeline
from rag_eval_kit.safety import redact_pii

Role = Literal["reader", "admin"]


@dataclass(frozen=True)
class ApiKey:
    """A client identity. Only the SHA-256 digest of the secret is kept in memory."""

    name: str
    role: Role
    digest: str


def _digest(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def parse_api_keys(spec: str) -> list[ApiKey]:
    """Parse ``name:role:secret`` entries separated by commas (the ``RAG_API_KEYS`` format)."""
    keys: list[ApiKey] = []
    for entry in filter(None, (part.strip() for part in spec.split(","))):
        parts = entry.split(":", 2)
        if len(parts) != 3 or parts[1] not in ("reader", "admin") or len(parts[2]) < 16:
            raise ValueError(
                "each API key entry must be name:role:secret with role reader or admin and a secret of 16+ characters"
            )
        keys.append(ApiKey(parts[0], parts[1], _digest(parts[2])))  # type: ignore[arg-type]
    return keys


def keys_from_env() -> list[ApiKey]:
    """Read API keys from ``RAG_API_KEYS``. There is no default key."""
    keys = parse_api_keys(os.environ.get("RAG_API_KEYS", ""))
    if not keys:
        raise RuntimeError("RAG_API_KEYS is not set. Refusing to start without API keys.")
    return keys


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    mode: Literal["bm25", "dense", "hybrid"] | None = None
    rerank: bool | None = None


class CitationOut(BaseModel):
    marker: int
    chunk_id: str
    doc_id: str
    title: str
    section: str


class QueryResponse(BaseModel):
    request_id: str
    answer: str
    refused: bool
    blocked: bool
    citations: list[CitationOut]
    flags: list[str]
    latency_ms: float


class HitOut(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    section: str
    score: float
    text: str


class SearchResponse(BaseModel):
    request_id: str
    hits: list[HitOut]


class DocumentIn(BaseModel):
    doc_id: str = Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9._-]+$")
    title: str = Field(min_length=1, max_length=500)
    text: str = Field(min_length=1, max_length=200_000)


class IngestRequest(BaseModel):
    documents: list[DocumentIn] = Field(min_length=1, max_length=500)


def create_app(pipeline: RagPipeline, api_keys: list[ApiKey], audit: AuditLog | None = None) -> FastAPI:
    """Build the application around an ingested pipeline."""
    if not api_keys:
        raise ValueError("at least one API key is required")
    log = audit if audit is not None else AuditLog()
    app = FastAPI(title="rag-eval-kit", version="0.1.0")
    app.state.audit = log

    def authenticate(x_api_key: str | None = Header(default=None)) -> ApiKey:
        presented = _digest(x_api_key or "")
        match = None
        for key in api_keys:  # compare against every key to keep timing independent of position
            if hmac.compare_digest(key.digest, presented):
                match = key
        if match is None:
            log.append("auth_failed", reason="missing key" if not x_api_key else "unknown key")
            raise HTTPException(status_code=401, detail="invalid or missing API key")
        return match

    def require_admin(key: ApiKey = Depends(authenticate)) -> ApiKey:
        if key.role != "admin":
            log.append("forbidden", client=key.name, required_role="admin")
            raise HTTPException(status_code=403, detail="admin role required")
        return key

    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        return {"status": "ok", "documents": len(pipeline.documents), "chunks": len(pipeline.chunks)}

    @app.post("/v1/query", response_model=QueryResponse)
    def query(request: QueryRequest, response: Response, key: ApiKey = Depends(authenticate)) -> QueryResponse:
        request_id = str(uuid.uuid4())
        result = pipeline.answer(request.question, request.top_k, request.mode, request.rerank)
        log.append(
            "query",
            request_id=request_id,
            client=key.name,
            question=redact_pii(request.question),
            retrieved=[hit.chunk.chunk_id for hit in result.hits],
            cited=[c.chunk_id for c in result.answer.citations],
            refused=result.answer.refused,
            blocked=result.blocked,
            flags=list(result.flags),
            latency_ms=round(result.latency_ms, 3),
        )
        response.headers["X-Request-ID"] = request_id
        return QueryResponse(
            request_id=request_id,
            answer=result.answer.text,
            refused=result.answer.refused,
            blocked=result.blocked,
            citations=[CitationOut(**vars(c)) for c in result.answer.citations],
            flags=list(result.flags),
            latency_ms=round(result.latency_ms, 3),
        )

    @app.post("/v1/search", response_model=SearchResponse)
    def search(request: QueryRequest, key: ApiKey = Depends(authenticate)) -> SearchResponse:
        request_id = str(uuid.uuid4())
        hits = pipeline.search(request.question, request.top_k, request.mode, request.rerank)
        log.append(
            "search",
            request_id=request_id,
            client=key.name,
            question=redact_pii(request.question),
            retrieved=[hit.chunk.chunk_id for hit in hits],
        )
        out = [
            HitOut(
                chunk_id=h.chunk.chunk_id,
                doc_id=h.chunk.doc_id,
                title=h.chunk.title,
                section=h.chunk.section,
                score=round(h.score, 6),
                text=h.chunk.text,
            )
            for h in hits
        ]
        return SearchResponse(request_id=request_id, hits=out)

    @app.post("/v1/documents", status_code=201)
    def ingest(request: IngestRequest, key: ApiKey = Depends(require_admin)) -> dict[str, object]:
        chunks = pipeline.ingest([Document(d.doc_id, d.title, d.text) for d in request.documents])
        log.append("ingest", client=key.name, doc_ids=[d.doc_id for d in request.documents], chunks=chunks)
        return {"documents": len(pipeline.documents), "chunks": chunks, "quarantined": sorted(pipeline.quarantined)}

    @app.get("/v1/audit/verify")
    def verify_audit(key: ApiKey = Depends(require_admin)) -> dict[str, object]:
        valid, first_bad = log.verify()
        return {"valid": valid, "records": len(log), "first_invalid_seq": first_bad}

    return app
