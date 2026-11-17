"""LLM provider interface, a deterministic offline mock and an optional OpenAI adapter."""

from __future__ import annotations

import os
import re
from typing import Protocol

from pydantic import BaseModel

from .embedding import estimate_tokens, tokenize


class Completion(BaseModel):
    """Provider-independent completion result."""

    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str
    latency_simulated: bool = False


class LLMProvider(Protocol):
    def complete(self, prompt: str, model: str) -> Completion: ...


class KeywordMockLLM:
    """Deterministic stand-in for an LLM used as a zero-shot classifier.

    It is NOT a language model. It reads the label catalogue out of the prompt
    (lines of the form ``- label: description``), reads the text after the last
    ``Ticket:`` marker, and answers with the label whose name and description
    share the most words with the ticket. A label that is missing from the
    prompt can never be predicted, and a label without a description only
    matches on its own name, so prompt edits change accuracy for real, causal
    reasons. That is all the evaluation gate needs in order to be demonstrated.

    Latency is simulated with a linear token model and flagged as such.
    """

    def __init__(self, base_latency_ms: float = 120.0, ms_per_token: float = 0.35) -> None:
        self.base_latency_ms = base_latency_ms
        self.ms_per_token = ms_per_token

    def complete(self, prompt: str, model: str) -> Completion:
        labels = re.findall(r"^- ([a-z_]+)\s*:?\s*(.*)$", prompt, re.M)
        ticket = set(tokenize(prompt.rsplit("Ticket:", 1)[-1]))
        best, best_score = "unknown", 0
        for label, description in labels:
            score = len(ticket & (set(tokenize(description)) | {label}))
            if score > best_score:
                best, best_score = label, score
        if best_score == 0 and labels:
            best = labels[0][0]
        input_tokens, output_tokens = estimate_tokens(prompt), estimate_tokens(best)
        return Completion(
            text=best,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=round(self.base_latency_ms + self.ms_per_token * input_tokens, 3),
            model=model,
            latency_simulated=True,
        )


class OpenAIChatProvider:
    """Thin adapter for Azure OpenAI or OpenAI chat completions.

    UNTESTED in this repository: it requires network access and credentials that
    the offline authoring environment does not have. Environment variables:
    ``AZURE_OPENAI_ENDPOINT``, ``AZURE_OPENAI_API_KEY``, ``AZURE_OPENAI_API_VERSION``
    for Azure (``model`` is then the deployment name), or ``OPENAI_API_KEY``.
    """

    def __init__(self) -> None:
        try:
            import openai  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install the 'openai' extra to use OpenAIChatProvider") from exc
        if os.environ.get("AZURE_OPENAI_ENDPOINT"):  # pragma: no cover - needs network
            self._client = openai.AzureOpenAI(
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01"),
            )
        else:  # pragma: no cover - needs network
            self._client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def complete(self, prompt: str, model: str) -> Completion:  # pragma: no cover
        import time

        start = time.perf_counter()
        result = self._client.chat.completions.create(
            model=model, temperature=0.0, messages=[{"role": "user", "content": prompt}]
        )
        usage = result.usage
        return Completion(
            text=(result.choices[0].message.content or "").strip(),
            input_tokens=usage.prompt_tokens if usage else estimate_tokens(prompt),
            output_tokens=usage.completion_tokens if usage else 1,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            model=model,
        )
