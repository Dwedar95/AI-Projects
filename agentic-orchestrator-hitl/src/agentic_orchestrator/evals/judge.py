"""LLM-as-judge interface.

Deterministic checks decide pass or fail. The judge adds a graded quality score
for answers where string checks are too coarse. In this repository the judge is
only ever driven by the offline mock, whose "judgement" is reference-term
recall, so the reported judge scores validate the plumbing and not the quality
of a real judge model.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from ..guardrails import StructuredOutputError, structured_completion
from ..llm import LLMProvider
from ..models import Message

JUDGE_SYSTEM = (
    "ROLE: judge\n"
    "Grade how well ANSWER satisfies QUESTION given REFERENCE. "
    'Reply with JSON: {"score": number between 0 and 1, "rationale": "one sentence"}.'
)


class JudgeVerdict(BaseModel):
    """Structured judge output."""

    score: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class Judge(Protocol):
    def grade(self, question: str, reference: str, answer: str) -> JudgeVerdict: ...


class LLMJudge:
    """Judge backed by any ``LLMProvider`` with schema-validated verdicts."""

    def __init__(self, llm: LLMProvider, max_retries: int = 1) -> None:
        self.llm = llm
        self.max_retries = max_retries

    def grade(self, question: str, reference: str, answer: str) -> JudgeVerdict:
        def flat(text: str) -> str:
            return " ".join(text.split())

        messages = [
            Message(role="system", content=JUDGE_SYSTEM),
            Message(
                role="user",
                content=(
                    f"QUESTION: {flat(question)}\nREFERENCE: {flat(reference)}\n"
                    f"ANSWER: {flat(answer)}"
                ),
            ),
        ]
        try:
            verdict, _ = structured_completion(self.llm, messages, JudgeVerdict, self.max_retries)
        except StructuredOutputError as exc:
            return JudgeVerdict(score=0.0, rationale=f"judge output invalid: {exc}")
        return verdict
