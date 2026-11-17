"""Conversation memory and run state persistence."""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .llm import estimate_tokens
from .models import RunState


@dataclass
class _Session:
    turns: list[tuple[str, str]] = field(default_factory=list)
    summary: list[str] = field(default_factory=list)


class ConversationMemory:
    """Per-session memory with a token budget and extractive roll-up.

    Recent turns are kept verbatim. When the budget is exceeded the oldest turns
    are folded into a summary that keeps the first sentence of each turn. The
    roll-up is deterministic and extractive on purpose: an abstractive LLM
    summary would be a better product choice but would make runs irreproducible.
    """

    def __init__(self, max_tokens: int = 300, max_summary_items: int = 8) -> None:
        if max_tokens < 20:
            raise ValueError("max_tokens must be at least 20")
        self.max_tokens = max_tokens
        self.max_summary_items = max_summary_items
        self._sessions: dict[str, _Session] = {}

    def add(self, session_id: str, role: str, content: str) -> None:
        """Append a turn and evict old turns into the summary if over budget."""
        session = self._sessions.setdefault(session_id, _Session())
        session.turns.append((role, content.strip()))
        while len(session.turns) > 1 and self._tokens(session) > self.max_tokens:
            old_role, old_content = session.turns.pop(0)
            first = re.split(r"(?<=[.!?])\s", old_content, maxsplit=1)[0][:160]
            session.summary.append(f"{old_role}: {first}")
            session.summary = session.summary[-self.max_summary_items :]

    @staticmethod
    def _tokens(session: _Session) -> int:
        return sum(estimate_tokens(content) for _, content in session.turns)

    def turns(self, session_id: str) -> list[tuple[str, str]]:
        return list(self._sessions.get(session_id, _Session()).turns)

    def context(self, session_id: str | None) -> str:
        """Render memory for inclusion in a prompt. Empty string for unknown sessions."""
        if session_id is None or session_id not in self._sessions:
            return ""
        session = self._sessions[session_id]
        lines: list[str] = []
        if session.summary:
            lines.append("Earlier (summarised): " + " | ".join(session.summary))
        lines.extend(f"{role}: {content}" for role, content in session.turns)
        return "\n".join(lines)


class StateStore(Protocol):
    """Persistence contract for run state."""

    def save(self, state: RunState) -> None: ...

    def get(self, run_id: str) -> RunState | None: ...

    def list(self) -> list[RunState]: ...


class InMemoryStateStore:
    """Stores deep copies so callers cannot mutate persisted state by accident."""

    def __init__(self) -> None:
        self._states: dict[str, RunState] = {}

    def save(self, state: RunState) -> None:
        self._states[state.run_id] = state.model_copy(deep=True)

    def get(self, run_id: str) -> RunState | None:
        state = self._states.get(run_id)
        return state.model_copy(deep=True) if state else None

    def list(self) -> list[RunState]:
        return [s.model_copy(deep=True) for s in self._states.values()]


class JsonFileStateStore:
    """One JSON file per run, written atomically (temp file plus rename)."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_]+", run_id):
            raise ValueError("invalid run id")
        return self.directory / f"{run_id}.json"

    def save(self, state: RunState) -> None:
        fd, tmp = tempfile.mkstemp(dir=self.directory, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(state.model_dump_json())
        os.replace(tmp, self._path(state.run_id))

    def get(self, run_id: str) -> RunState | None:
        path = self._path(run_id)
        if not path.exists():
            return None
        return RunState.model_validate_json(path.read_text(encoding="utf-8"))

    def list(self) -> list[RunState]:
        return [
            RunState.model_validate_json(p.read_text(encoding="utf-8"))
            for p in sorted(self.directory.glob("*.json"))
        ]
