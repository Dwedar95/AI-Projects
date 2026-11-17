"""Chunking strategies.

* ``fixed``: sliding window over words with overlap. Simple, ignores structure.
* ``sentence``: packs whole sentences up to a word budget, with sentence overlap.
* ``heading``: one chunk per markdown section, long sections are split by sentence.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from rag_eval_kit.documents import Chunk, Document
from rag_eval_kit.text import split_sentences

_HEADING = re.compile(r"^#{2,6}\s+(.*)$")


class Chunker(Protocol):
    """A chunking strategy."""

    def chunk(self, document: Document) -> list[Chunk]: ...


def _make_chunks(document: Document, pieces: list[tuple[str, str]]) -> list[Chunk]:
    return [
        Chunk(f"{document.doc_id}::{i}", document.doc_id, document.title, section, text, i)
        for i, (section, text) in enumerate(pieces)
        if text.strip()
    ]


def _flatten(text: str) -> str:
    """Remove heading markers and collapse whitespace."""
    lines = [line.lstrip("#").strip() for line in text.splitlines()]
    return " ".join(line for line in lines if line)


@dataclass(frozen=True)
class FixedWindowChunker:
    """Sliding window of ``size`` words that advances by ``size - overlap`` words."""

    size: int = 80
    overlap: int = 20

    def __post_init__(self) -> None:
        if self.size <= 0 or not 0 <= self.overlap < self.size:
            raise ValueError("size must be positive and overlap must be in [0, size)")

    def chunk(self, document: Document) -> list[Chunk]:
        tokens = _flatten(document.text).split()
        step = self.size - self.overlap
        pieces = []
        for start in range(0, max(len(tokens), 1), step):
            pieces.append(("", " ".join(tokens[start : start + self.size])))
            if start + self.size >= len(tokens):
                break
        return _make_chunks(document, pieces)


def pack_sentences(sentences: list[str], max_words: int, overlap_sentences: int) -> list[str]:
    """Greedily pack sentences into groups of at most ``max_words`` words.

    A sentence longer than the budget becomes its own group. Consecutive groups
    share ``overlap_sentences`` trailing sentences.
    """
    groups: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in sentences:
        length = len(sentence.split())
        if current and count + length > max_words:
            groups.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            count = sum(len(s.split()) for s in current)
        current.append(sentence)
        count += length
    if current:
        groups.append(" ".join(current))
    return groups


@dataclass(frozen=True)
class SentenceChunker:
    """Sentence-aligned chunks of at most ``max_words`` words."""

    max_words: int = 80
    overlap_sentences: int = 1

    def chunk(self, document: Document) -> list[Chunk]:
        groups = pack_sentences(split_sentences(document.text), self.max_words, self.overlap_sentences)
        return _make_chunks(document, [("", group) for group in groups])


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into ``(heading, body)`` pairs. Text before a heading has an empty heading."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in text.splitlines():
        match = _HEADING.match(line)
        if match:
            sections.append((match.group(1).strip(), []))
        else:
            sections[-1][1].append(line)
    return [(heading, "\n".join(body).strip()) for heading, body in sections if "\n".join(body).strip()]


@dataclass(frozen=True)
class HeadingChunker:
    """One chunk per markdown section. Sections above ``max_words`` are split by sentence."""

    max_words: int = 120
    overlap_sentences: int = 1

    def chunk(self, document: Document) -> list[Chunk]:
        pieces: list[tuple[str, str]] = []
        for heading, body in split_sections(document.text):
            for group in pack_sentences(split_sentences(body), self.max_words, self.overlap_sentences):
                pieces.append((heading, group))
        return _make_chunks(document, pieces)


CHUNKERS: dict[str, Callable[[], Chunker]] = {
    "fixed": FixedWindowChunker,
    "sentence": SentenceChunker,
    "heading": HeadingChunker,
}


def get_chunker(name: str) -> Chunker:
    """Return a chunker by strategy name."""
    if name not in CHUNKERS:
        raise ValueError(f"unknown chunking strategy '{name}', choose from {sorted(CHUNKERS)}")
    return CHUNKERS[name]()
