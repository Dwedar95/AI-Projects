"""LLM provider interface plus a replay provider and an optional OpenAI adapter.

The orchestrator only depends on :class:`LLMProvider`. Tests and demos use the
deterministic providers (``ScriptedLLM`` here, ``MockLLM`` in ``mock_llm.py``),
so nothing in this repository needs network access.
"""

from __future__ import annotations

import os
from collections import deque
from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from .models import Message


class LLMResponse(BaseModel):
    """Provider-independent completion result."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = "unknown"


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal contract every model backend has to satisfy."""

    def complete(
        self, messages: list[Message], json_schema: dict[str, Any] | None = None
    ) -> LLMResponse:
        """Return a completion. When ``json_schema`` is given the text should be JSON."""
        ...


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (about four characters per token) for offline accounting."""
    return max(1, len(text) // 4)


class ScriptedLLM:
    """Replays a fixed list of responses in order. Used to script exact agent traces."""

    def __init__(self, responses: Iterable[str]) -> None:
        self._responses: deque[str] = deque(responses)
        self.calls: list[list[Message]] = []

    def complete(
        self, messages: list[Message], json_schema: dict[str, Any] | None = None
    ) -> LLMResponse:
        self.calls.append(list(messages))
        if not self._responses:
            raise RuntimeError("ScriptedLLM has no responses left")
        text = self._responses.popleft()
        return LLMResponse(
            text=text,
            prompt_tokens=sum(estimate_tokens(m.content) for m in messages),
            completion_tokens=estimate_tokens(text),
            model="scripted",
        )


class OpenAIChatLLM:
    """Thin adapter for OpenAI and Azure OpenAI chat completions.

    UNTESTED in this repository: it needs network access and credentials, which
    the offline test environment does not have. Configuration comes from the
    environment:

    - Azure: ``AZURE_OPENAI_ENDPOINT``, ``AZURE_OPENAI_API_KEY``,
      ``AZURE_OPENAI_DEPLOYMENT`` and optionally ``AZURE_OPENAI_API_VERSION``.
    - OpenAI: ``OPENAI_API_KEY`` and optionally ``OPENAI_MODEL``.
    """

    def __init__(self, temperature: float = 0.0) -> None:
        try:
            import openai  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install the 'openai' extra to use OpenAIChatLLM") from exc
        self.temperature = temperature
        if os.environ.get("AZURE_OPENAI_ENDPOINT"):  # pragma: no cover - needs network
            self._client = openai.AzureOpenAI(
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01"),
            )
            self._model = os.environ["AZURE_OPENAI_DEPLOYMENT"]
        else:  # pragma: no cover - needs network
            self._client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            self._model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    def complete(
        self, messages: list[Message], json_schema: dict[str, Any] | None = None
    ) -> LLMResponse:  # pragma: no cover - needs network
        kwargs: dict[str, Any] = {}
        if json_schema is not None:
            kwargs["response_format"] = {"type": "json_object"}
        result = self._client.chat.completions.create(
            model=self._model,
            temperature=self.temperature,
            messages=[m.model_dump() for m in messages],
            **kwargs,
        )
        usage = result.usage
        return LLMResponse(
            text=result.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            model=self._model,
        )
