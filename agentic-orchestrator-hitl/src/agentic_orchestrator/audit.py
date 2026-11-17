"""Append-only, hash-chained audit log.

Every record stores the SHA-256 of the previous record, so editing, removing or
reordering a line breaks the chain and is detected by :meth:`AuditLog.verify`.
Records are written as JSON lines and the file is only ever opened in append
mode. This gives tamper evidence, not tamper prevention: someone with write
access can still rewrite the whole file with a consistent chain, so production
use needs the head hash anchored somewhere the writer cannot change.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

GENESIS_HASH = "0" * 64


class AuditRecord(BaseModel):
    """One immutable audit entry."""

    seq: int
    ts: float
    run_id: str
    event: str
    actor: str
    payload: dict[str, Any]
    prev_hash: str
    hash: str = ""

    def compute_hash(self) -> str:
        body = self.model_dump(exclude={"hash"})
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class AuditLog:
    """Hash-chained event log kept in memory and optionally mirrored to a JSONL file."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._records: list[AuditRecord] = []
        self._lock = threading.Lock()
        if self.path is not None and self.path.exists():
            with self.path.open(encoding="utf-8") as handle:
                self._records = [
                    AuditRecord.model_validate_json(line) for line in handle if line.strip()
                ]

    def append(self, run_id: str, event: str, actor: str, payload: dict[str, Any]) -> AuditRecord:
        """Add a record to the end of the chain and persist it."""
        with self._lock:
            prev_hash = self._records[-1].hash if self._records else GENESIS_HASH
            record = AuditRecord(
                seq=len(self._records),
                ts=time.time(),
                run_id=run_id,
                event=event,
                actor=actor,
                payload=json.loads(json.dumps(payload, default=str)),
                prev_hash=prev_hash,
            )
            record.hash = record.compute_hash()
            self._records.append(record)
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(record.model_dump_json() + "\n")
            return record

    def records(self, run_id: str | None = None, event: str | None = None) -> list[AuditRecord]:
        """Return records, optionally filtered by run and event type."""
        return [
            r
            for r in self._records
            if (run_id is None or r.run_id == run_id) and (event is None or r.event == event)
        ]

    def verify(self) -> tuple[bool, str]:
        """Check sequence numbers, hashes and chain links. Returns ``(ok, detail)``."""
        prev_hash = GENESIS_HASH
        for index, record in enumerate(self._records):
            if record.seq != index:
                return False, f"record {index}: sequence number is {record.seq}"
            if record.prev_hash != prev_hash:
                return False, f"record {index}: chain link does not match previous hash"
            if record.compute_hash() != record.hash:
                return False, f"record {index}: content hash mismatch"
            prev_hash = record.hash
        return True, f"{len(self._records)} records verified"

    def __len__(self) -> int:
        return len(self._records)
