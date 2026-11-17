"""Safety checks: personal data detection and redaction, prompt injection detection.

Both detectors are pattern-based and deterministic. They are a first line of
defence that is cheap to run on every request, not a complete solution: they
miss paraphrased attacks and unusual PII formats. The evaluation reports their
measured precision and recall on the bundled synthetic probes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+\b")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,4})?\b")
_CARD = re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])")
_IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
_PHONE = re.compile(
    r"(?<![\w.])(?:"
    r"(?:\+|00)\d{1,3}[ -]?(?:\(?\d{1,4}\)?[ -]?){1,4}\d{2,4}"  # international: +49 30 5550 1234
    r"|\(?0\d{2,4}\)?[ /-]\d{3,4}[ -]?\d{3,5}"  # national: 030 5550 5678
    r")(?!\w)"
)


@dataclass(frozen=True)
class PiiMatch:
    kind: str
    start: int
    end: int
    value: str


def _luhn_valid(number: str) -> bool:
    digits = [int(ch) for ch in number if ch.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2 == 1:
            digit = digit * 2 - 9 if digit > 4 else digit * 2
        total += digit
    return total % 10 == 0


def _iban_valid(candidate: str) -> bool:
    compact = candidate.replace(" ", "")
    if not 15 <= len(compact) <= 34:
        return False
    rearranged = compact[4:] + compact[:4]
    numeric = "".join(str(int(ch, 36)) for ch in rearranged)
    return int(numeric) % 97 == 1


def _ipv4_valid(candidate: str) -> bool:
    return all(0 <= int(part) <= 255 for part in candidate.split("."))


def _phone_valid(candidate: str) -> bool:
    return 8 <= sum(ch.isdigit() for ch in candidate) <= 15


_DETECTORS = (
    ("EMAIL", _EMAIL, lambda _value: True),
    ("IBAN", _IBAN, _iban_valid),
    ("CREDIT_CARD", _CARD, _luhn_valid),
    ("IP_ADDRESS", _IPV4, _ipv4_valid),
    ("PHONE", _PHONE, _phone_valid),
)


def find_pii(text: str) -> list[PiiMatch]:
    """Return non-overlapping PII matches. Earlier detectors win on overlap."""
    matches: list[PiiMatch] = []
    for kind, pattern, validator in _DETECTORS:
        for match in pattern.finditer(text):
            value = match.group().strip()
            overlaps = any(match.start() < m.end and m.start < match.end() for m in matches)
            if not overlaps and validator(value):
                matches.append(PiiMatch(kind, match.start(), match.start() + len(value), value))
    return sorted(matches, key=lambda m: m.start)


def redact_pii(text: str) -> str:
    """Replace every detected PII value with a ``[KIND]`` placeholder."""
    result, cursor = [], 0
    for match in find_pii(text):
        result.append(text[cursor : match.start])
        result.append(f"[{match.kind}]")
        cursor = match.end
    result.append(text[cursor:])
    return "".join(result)


_GAP = r"[^.?!\n]"
_INJECTION_SPECS: tuple[tuple[str, float, str], ...] = (
    (
        "override_instructions",
        0.9,
        rf"\b(ignore|disregard|forget|override)\b{_GAP}{{0,40}}\b(previous|prior|above|earlier|initial|your|all)\b"
        rf"{_GAP}{{0,30}}\b(instructions?|prompts?|rules|context|guidelines)\b",
    ),
    (
        "reveal_system_prompt",
        0.9,
        rf"\b(reveal|show|print|repeat|output|leak|display)\b{_GAP}{{0,40}}\b(system|hidden|initial|original)\b"
        rf"{_GAP}{{0,20}}\b(prompt|instructions?|message)\b",
    ),
    (
        "role_hijack",
        0.7,
        rf"\b(you are now|pretend to be|act as (if|an?))\b{_GAP}{{0,60}}"
        r"\b(unrestricted|no restrictions|without (any )?(rules|guardrails|restrictions)|dan)\b",
    ),
    ("jailbreak_terms", 0.7, r"\b(developer mode|jailbreak|do anything now|DAN)\b"),
    (
        "bypass_safety",
        0.8,
        rf"\b(bypass|override|disable|turn off)\b{_GAP}{{0,30}}"
        r"\b(safety|security|guardrails?|content filters?|filters?|polic(y|ies))\b",
    ),
    ("fake_system_turn", 0.7, r"(^|\n)\s*(#+\s*)?(system|assistant)\s*:|<\s*/?\s*system\s*>|\bnew instructions?\s*:"),
    (
        "suppress_citations",
        0.5,
        rf"\b(do not|don't|never)\b{_GAP}{{0,20}}\b(cite|mention|reference)\b{_GAP}{{0,20}}\bsources?\b",
    ),
    (
        "credential_exfiltration",
        0.6,
        rf"\b(reveal|output|print|show|tell me|send)\b{_GAP}{{0,40}}"
        r"\b(api keys?|passwords?|credentials|secrets?|tokens?)\b",
    ),
    ("forced_output", 0.5, rf"\b(respond|reply|answer)\b{_GAP}{{0,15}}\bonly with\b"),
)
# Weights of all matching patterns combine as 1 - prod(1 - w).
_INJECTION_PATTERNS = tuple(
    (name, weight, re.compile(pattern, re.IGNORECASE)) for name, weight, pattern in _INJECTION_SPECS
)


@dataclass(frozen=True)
class InjectionResult:
    score: float
    matched: tuple[str, ...]
    threshold: float = 0.5

    @property
    def flagged(self) -> bool:
        return self.score >= self.threshold


def detect_injection(text: str, threshold: float = 0.5) -> InjectionResult:
    """Score a user message or a retrieved passage for prompt injection patterns."""
    remaining = 1.0
    matched: list[str] = []
    for name, weight, pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            matched.append(name)
            remaining *= 1.0 - weight
    return InjectionResult(round(1.0 - remaining, 4), tuple(matched), threshold)
