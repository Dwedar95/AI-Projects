"""Tokenisation helpers shared by the sparse index, the embedders and the metrics."""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\[])")

STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "do", "does", "for", "from",
    "had", "has", "have", "how", "i", "if", "in", "into", "is", "it", "its", "may", "me", "my", "no", "not",
    "of", "on", "or", "our", "shall", "should", "so", "than", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "to", "was", "we", "were", "what", "when", "where", "which", "who",
    "will", "with", "would", "you", "your",
})  # fmt: skip


def stem(token: str) -> str:
    """Light suffix stripping. Keeps identifiers such as ``err-4012`` untouched."""
    if any(ch.isdigit() for ch in token) or len(token) <= 3:
        return token
    for suffix, replacement in (("ies", "y"), ("sses", "ss"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            if suffix == "s" and token.endswith("ss"):
                return token
            return token[: len(token) - len(suffix)] + replacement
    return token


def words(text: str) -> list[str]:
    """Lowercased word tokens, including compound identifiers, without filtering."""
    return _WORD.findall(text.lower())


def tokenize(text: str) -> list[str]:
    """Tokens for indexing and matching: lowercased, stopwords removed, stemmed."""
    return [stem(token) for token in words(text) if token not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences. List items and headings become their own sentences."""
    sentences: list[str] = []
    for line in text.splitlines():
        line = line.strip().lstrip("#").strip()
        if not line:
            continue
        line = re.sub(r"^[-*]\s+", "", line)
        sentences.extend(part.strip() for part in _SENTENCE_END.split(line) if part.strip())
    return sentences
