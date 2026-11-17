"""LLM provider abstraction with a deterministic offline mock and an optional HTTP adapter.

The engine only talks to :class:`LLMProvider`. Two hosting kinds are modelled:
``proprietary`` (a managed API such as Azure OpenAI) and ``open_source`` (a
self-hosted open-weight model behind an OpenAI-compatible endpoint). Tests and
demos use :class:`MockProvider`, which implements each task with documented
heuristics so that routing, cascading, confidence gates and cost accounting can
be exercised without a network. It is not a language model.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from .retriever import stem, tokenize

Hosting = Literal["proprietary", "open_source"]
Tier = Literal["cheap", "strong"]


@dataclass(frozen=True)
class ModelSpec:
    """A model entry of the catalog. Prices are per 1000 tokens."""

    name: str
    hosting: Hosting
    tier: Tier
    price_in_per_1k: float
    price_out_per_1k: float
    base_latency_ms: float
    ms_per_output_token: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.price_in_per_1k + output_tokens * self.price_out_per_1k) / 1000.0


@dataclass
class LLMRequest:
    """A task request. ``prompt`` is the instruction, ``text`` the payload it applies to."""

    task: str
    prompt: str
    text: str = ""
    labels: dict[str, list[str]] = field(default_factory=dict)
    fields: dict[str, str] = field(default_factory=dict)
    passages: list[tuple[str, str]] = field(default_factory=list)
    max_output_tokens: int = 120


@dataclass
class LLMResponse:
    """Provider answer with structured data, a confidence in 0..1 and usage."""

    text: str
    data: dict[str, Any]
    confidence: float
    input_tokens: int
    output_tokens: int
    model: str
    latency_ms: float


class LLMProvider(Protocol):
    """Interface every provider implements."""

    def complete(self, model: ModelSpec, request: LLMRequest) -> LLMResponse: ...


def count_tokens(text: str) -> int:
    """Approximate token count: words and punctuation marks. Used for cost estimates and mock usage."""
    return len(re.findall(r"\w+|[^\w\s]", text))


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _truncate(text: str, max_tokens: int) -> str:
    words = text.split()
    return text if len(words) <= max_tokens else " ".join(words[:max_tokens]) + " ..."


_AMOUNT = re.compile(
    r"(?:EUR|USD|\$|€)\s*(\d[\d.,]*\d|\d)|(\d[\d.,]*\d|\d)\s*(?:EUR|USD|euros?|dollars?|€)", re.IGNORECASE
)
_SIMPLE_FIELDS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "date": re.compile(r"\d{4}-\d{2}-\d{2}"),
    "integer": re.compile(r"\b\d+\b"),
}


class MockProvider:
    """Deterministic stand-in for a model. Same input always gives the same output.

    Tier differences are deliberate and documented so that cascading has an effect:
    - classify: the cheap tier matches keywords as exact tokens, the strong tier
      also matches inflected forms (stemming) on both sides;
    - extract: the cheap tier reads plain amounts (``120.50``), the strong tier
      also reads thousands separators (``1,250.00``);
    - summarise and grounded answers: one sentence (cheap) or two (strong).
    """

    def complete(self, model: ModelSpec, request: LLMRequest) -> LLMResponse:
        strong = model.tier == "strong"
        handler = {
            "classify": self._classify, "extract": self._extract,
            "summarise": self._summarise, "generate": self._generate,
        }.get(request.task)
        if handler is None:
            raise ValueError(f"unsupported task: {request.task}")
        text, data, confidence = handler(request, strong)
        passages = " ".join(p for _, p in request.passages)
        input_tokens = count_tokens(request.prompt) + count_tokens(request.text) + count_tokens(passages)
        output_tokens = count_tokens(text)
        latency = model.base_latency_ms + model.ms_per_output_token * output_tokens
        return LLMResponse(text, data, round(confidence, 3), input_tokens, output_tokens, model.name, latency)

    @staticmethod
    def _classify(req: LLMRequest, strong: bool) -> tuple[str, dict[str, Any], float]:
        lowered = " " + " ".join(re.findall(r"[a-z0-9]+", req.text.lower())) + " "
        tokens = set(tokenize(req.text, stemming=strong))
        scores: dict[str, int] = {}
        for label, keywords in req.labels.items():
            hits = 0
            for kw in dict.fromkeys(keywords):
                kw = kw.lower()
                if " " in kw:
                    hits += f" {kw} " in lowered
                else:
                    hits += (stem(kw) if strong else kw) in tokens
            scores[label] = hits
        ranked = sorted(scores, key=lambda k: -scores[k])  # stable: declared order breaks ties
        top = scores[ranked[0]]
        second = scores[ranked[1]] if len(ranked) > 1 else 0
        label = ranked[0] if top > 0 else "unknown"
        confidence = 0.5 + 0.5 * (top - second) / top if top > 0 else 0.0
        return json.dumps({"label": label}), {"label": label, "scores": scores}, confidence

    @staticmethod
    def _extract(req: LLMRequest, strong: bool) -> tuple[str, dict[str, Any], float]:
        data: dict[str, Any] = {}
        for name, kind in req.fields.items():
            value: Any = None
            if kind == "amount":
                m = _AMOUNT.search(req.text)
                if m:
                    raw = m.group(1) or m.group(2)
                    if re.fullmatch(r"\d+(\.\d{1,2})?", raw):
                        value = float(raw)
                    elif strong and re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d{1,2})?", raw):
                        value = float(raw.replace(",", ""))
            elif kind.startswith("regex:"):
                m = re.search(kind[len("regex:"):], req.text)
                if m:
                    value = m.group(1) if m.groups() else m.group(0)
            elif kind in _SIMPLE_FIELDS:
                m = _SIMPLE_FIELDS[kind].search(req.text)
                if m:
                    value = int(m.group(0)) if kind == "integer" else m.group(0)
            else:
                raise ValueError(f"unknown field type '{kind}' for field '{name}'")
            data[name] = value
        found = sum(v is not None for v in data.values())
        return json.dumps(data), data, found / len(data)

    @staticmethod
    def _summarise(req: LLMRequest, strong: bool) -> tuple[str, dict[str, Any], float]:
        sentences = _sentences(req.text)
        text = _truncate(" ".join(sentences[: 2 if strong else 1]), req.max_output_tokens)
        return text, {"summary": text}, 0.9 if sentences else 0.0

    @staticmethod
    def _generate(req: LLMRequest, strong: bool) -> tuple[str, dict[str, Any], float]:
        if not req.passages:
            text = "[mock draft] " + _truncate(" ".join(req.text.split()), req.max_output_tokens)
            return text, {"text": text}, 0.9
        question = set(tokenize(req.text))
        scored = []
        for order, (doc_id, passage) in enumerate(req.passages):
            for sentence in _sentences(passage):
                overlap = len(question & set(tokenize(sentence)))
                scored.append((overlap / len(question) if question else 0.0, -order, sentence, doc_id))
        scored.sort(key=lambda s: (-s[0], -s[1]))
        if not scored or scored[0][0] == 0:
            text = "No grounded answer found in the knowledge base."
            return text, {"answer": text, "sources": []}, 0.0
        chosen = [s for s in scored[: 2 if strong else 1] if s[0] > 0]
        sources = list(dict.fromkeys(s[3] for s in chosen))
        text = " ".join(f"{s[2]} [source: {s[3]}]" for s in chosen)
        return text, {"answer": text, "sources": sources}, scored[0][0]


def build_messages(request: LLMRequest) -> list[dict[str, str]]:
    """Render a request as chat messages for a real model, asking for JSON on structured tasks."""
    system = request.prompt
    if request.task == "classify":
        system += (f"\nAnswer with JSON: {{\"label\": one of {sorted(request.labels)}, "
                   "\"confidence\": number between 0 and 1}}.")
    elif request.task == "extract":
        system += f"\nAnswer with JSON with the keys {sorted(request.fields)} and \"confidence\". Use null if absent."
    elif request.passages:
        system += "\nAnswer only from the passages and cite their ids. Say so if they do not contain the answer."
    user = request.text
    if request.passages:
        user += "\n\nPassages:\n" + "\n".join(f"[{doc_id}] {text}" for doc_id, text in request.passages)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_completion(payload: dict[str, Any], request: LLMRequest, model: ModelSpec, latency_ms: float) -> LLMResponse:
    """Convert an OpenAI-style chat completion payload into an :class:`LLMResponse`."""
    text = payload["choices"][0]["message"]["content"] or ""
    usage = payload.get("usage", {})
    data: dict[str, Any] = {"text": text}
    confidence = 0.5  # models do not report calibrated confidence; structured tasks may self-report it
    if request.task in ("classify", "extract"):
        try:
            data = json.loads(text)
            confidence = float(data.pop("confidence", confidence))
        except (json.JSONDecodeError, TypeError, ValueError):
            data, confidence = {"raw": text}, 0.0
    elif request.passages:
        data = {"answer": text, "sources": [d for d, _ in request.passages if f"[{d}]" in text or d in text]}
    return LLMResponse(text, data, confidence, int(usage.get("prompt_tokens", count_tokens(text))),
                       int(usage.get("completion_tokens", count_tokens(text))), model.name, latency_ms)


class OpenAICompatibleProvider:
    """Thin adapter for Azure OpenAI, OpenAI or a self-hosted OpenAI-compatible server (vLLM, Ollama).

    UNTESTED against live endpoints: this repository was built and tested offline.
    Message building and response parsing are unit-tested, the HTTP call is not.

    Environment variables read by :meth:`from_env`:
    - Azure OpenAI: ``AZURE_OPENAI_ENDPOINT``, ``AZURE_OPENAI_API_KEY``, ``AZURE_OPENAI_API_VERSION``.
      The catalog model name is used as the deployment name.
    - OpenAI or self-hosted: ``OPENAI_BASE_URL`` (default ``https://api.openai.com/v1``), ``OPENAI_API_KEY``.
    """

    def __init__(self, url_template: str, headers: dict[str, str], timeout: float = 60.0):
        self.url_template = url_template
        self.headers = {"Content-Type": "application/json", **headers}
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> OpenAICompatibleProvider:
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        if endpoint:
            version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01")
            url = f"{endpoint.rstrip('/')}/openai/deployments/{{model}}/chat/completions?api-version={version}"
            return cls(url, {"api-key": os.environ["AZURE_OPENAI_API_KEY"]})
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        return cls(f"{base}/chat/completions", {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"})

    def complete(self, model: ModelSpec, request: LLMRequest) -> LLMResponse:
        import time

        body = {"model": model.name, "messages": build_messages(request), "max_tokens": request.max_output_tokens,
                "temperature": 0}
        req = urllib.request.Request(self.url_template.format(model=model.name), data=json.dumps(body).encode(),
                                     headers=self.headers, method="POST")
        start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode())
        return parse_completion(payload, request, model, (time.perf_counter() - start) * 1000.0)
