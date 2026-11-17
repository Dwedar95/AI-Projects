"""Guardrails: prompt-injection heuristics and schema-validated structured output.

Two independent mechanisms live here:

1. ``detect_injection`` / ``sanitize_payload`` inspect untrusted text (tool
   outputs, retrieved documents) for instruction-like content and strip it
   before it reaches the model context.
2. ``structured_completion`` forces model output through a Pydantic schema,
   first trying a local JSON repair and then asking the model to fix its own
   output with the validation error as feedback.

The injection patterns are heuristics. They catch common phrasing and will miss
paraphrased or encoded attacks, which is why the approval gate exists as a
second, independent layer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .llm import LLMProvider, LLMResponse
from .models import Message

T = TypeVar("T", bound=BaseModel)

# (name, weight, pattern). Weights combine as independent evidence: 1 - prod(1 - w).
_INJECTION_RULES: list[tuple[str, float, re.Pattern[str]]] = [
    (
        "override_instructions",
        0.6,
        re.compile(
            r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}"
            r"\b(previous|prior|above|earlier|all)\b[^.\n]{0,20}"
            r"\b(instructions?|prompts?|rules?|context)\b",
            re.I,
        ),
    ),
    ("system_impersonation", 0.4, re.compile(r"\bsystem\s+(override|prompt|message)\b", re.I)),
    (
        "role_reassignment",
        0.3,
        re.compile(r"\b(you are now|from now on you|new instructions?:|act as (an?|the) )", re.I),
    ),
    (
        "secret_request",
        0.5,
        re.compile(
            r"\b(reveal|print|show|leak|output)\b[^.\n]{0,40}"
            r"\b(system prompt|api[ _-]?key|password|credentials?|secrets?)\b",
            re.I,
        ),
    ),
    (
        "exfiltration",
        0.4,
        re.compile(r"\b(send|email|forward|upload|post)\b[^.\n]{0,60}\b[\w.+-]+@[\w-]+\.\w+", re.I),
    ),
    ("concealment", 0.3, re.compile(r"\bdo not (tell|inform|mention)\b", re.I)),
    ("hidden_markup", 0.2, re.compile(r"<!--|[​‌‍⁠]")),
]

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class InjectionFinding:
    """Result of scanning one piece of untrusted text."""

    score: float
    rules: list[str] = field(default_factory=list)
    threshold: float = 0.5

    @property
    def flagged(self) -> bool:
        return self.score >= self.threshold


def detect_injection(text: str, threshold: float = 0.5) -> InjectionFinding:
    """Score how strongly ``text`` looks like an instruction aimed at the model."""
    remaining = 1.0
    rules: list[str] = []
    for name, weight, pattern in _INJECTION_RULES:
        if pattern.search(text):
            remaining *= 1.0 - weight
            rules.append(name)
    return InjectionFinding(score=round(1.0 - remaining, 4), rules=rules, threshold=threshold)


def sanitize_text(text: str) -> str:
    """Remove every sentence that matches an injection rule, keep the rest."""
    kept = [
        s
        for s in _SENTENCE_SPLIT.split(text)
        if s and not any(p.search(s) for _, _, p in _INJECTION_RULES)
    ]
    return " ".join(kept).strip()


def _strings(payload: Any) -> list[str]:
    if isinstance(payload, str):
        return [payload]
    if isinstance(payload, dict):
        return [s for v in payload.values() for s in _strings(v)]
    if isinstance(payload, list):
        return [s for v in payload for s in _strings(v)]
    return []


def scan_payload(payload: Any, threshold: float = 0.5) -> InjectionFinding:
    """Scan all string leaves of a JSON-like payload and return the worst finding."""
    worst = InjectionFinding(score=0.0, threshold=threshold)
    for text in _strings(payload):
        finding = detect_injection(text, threshold)
        if finding.score > worst.score:
            worst = finding
    return worst


def sanitize_payload(payload: Any) -> Any:
    """Apply :func:`sanitize_text` to every string leaf, preserving the structure."""
    if isinstance(payload, str):
        return sanitize_text(payload)
    if isinstance(payload, dict):
        return {k: sanitize_payload(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [sanitize_payload(v) for v in payload]
    return payload


class StructuredOutputError(Exception):
    """The model did not produce schema-valid output within the retry budget."""


@dataclass
class StructuredStats:
    """Bookkeeping for one structured completion."""

    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retries: int = 0
    local_repairs: int = 0

    def add(self, response: LLMResponse) -> None:
        self.llm_calls += 1
        self.prompt_tokens += response.prompt_tokens
        self.completion_tokens += response.completion_tokens


def _balanced_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` block, respecting JSON strings."""
    start = text.find("{")
    if start < 0:
        return None
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_json(text: str) -> tuple[dict[str, Any], bool]:
    """Parse a JSON object out of model text.

    Returns ``(data, repaired)``. ``repaired`` is True when the raw text was not
    valid JSON and had to be fixed locally (code fences, surrounding prose,
    trailing commas). Raises ``ValueError`` when no object can be recovered.
    """
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data, False
    except json.JSONDecodeError:
        pass
    candidate = _balanced_object(text)
    if candidate is None:
        raise ValueError("no complete JSON object found in model output")
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON is malformed: {exc.msg} at position {exc.pos}") from exc
    if not isinstance(data, dict):
        raise ValueError("top-level JSON value must be an object")
    return data, True


def structured_completion(
    llm: LLMProvider,
    messages: list[Message],
    schema: type[T],
    max_retries: int = 2,
) -> tuple[T, StructuredStats]:
    """Get schema-valid output from ``llm`` with repair and bounded retries.

    Each failed attempt is appended to the conversation together with the exact
    validation error, so the next attempt can correct it.
    """
    stats = StructuredStats()
    convo = list(messages)
    last_error = "unknown"
    for attempt in range(max_retries + 1):
        response = llm.complete(convo, json_schema=schema.model_json_schema())
        stats.add(response)
        try:
            data, repaired = extract_json(response.text)
            parsed = schema.model_validate(data)
        except (ValueError, ValidationError) as exc:
            last_error = str(exc).replace("\n", " ")[:400]
            if attempt < max_retries:
                stats.retries += 1
                convo = convo + [
                    Message(role="assistant", content=response.text),
                    Message(
                        role="user",
                        content=(
                            f"REPAIR: your previous output was rejected: {last_error}. "
                            "Reply with one JSON object that matches the schema and nothing else."
                        ),
                    ),
                ]
            continue
        if repaired:
            stats.local_repairs += 1
        return parsed, stats
    raise StructuredOutputError(
        f"no valid {schema.__name__} after {max_retries + 1} attempts: {last_error}"
    )
