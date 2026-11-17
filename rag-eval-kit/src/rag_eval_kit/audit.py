"""Append-only audit log with a hash chain.

Each record stores the SHA-256 of the previous record, so removing or editing
a line is detectable by :meth:`AuditLog.verify`. Queries are written after PII
redaction. The log is a JSON Lines file. Tamper evidence is not tamper
protection: an attacker with write access can rebuild the whole chain, so ship
the log to an append-only store in production.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

GENESIS = "0" * 64


def _digest(record: dict[str, Any]) -> str:
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class AuditLog:
    """Thread-safe hash-chained audit log. With ``path=None`` records stay in memory."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._records: list[dict[str, Any]] = []
        if self.path and self.path.exists():
            self._records = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line]

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> list[dict[str, Any]]:
        return list(self._records)

    def append(self, event: str, **fields: Any) -> dict[str, Any]:
        """Append a record and return it, including its hash."""
        with self._lock:
            record = {
                "seq": len(self._records) + 1,
                "ts": round(time.time(), 3),
                "event": event,
                **fields,
                "prev_hash": self._records[-1]["hash"] if self._records else GENESIS,
            }
            record["hash"] = _digest(record)
            self._records.append(record)
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
            return record

    def verify(self) -> tuple[bool, int | None]:
        """Check the chain. Returns ``(True, None)`` or ``(False, seq of the first bad record)``."""
        previous = GENESIS
        for index, record in enumerate(self._records, start=1):
            body = {k: v for k, v in record.items() if k != "hash"}
            if record.get("prev_hash") != previous or record.get("seq") != index or _digest(body) != record.get("hash"):
                return False, index
            previous = record["hash"]
        return True, None
