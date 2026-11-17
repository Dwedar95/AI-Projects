"""Time-series store backed by sqlite.

A narrow table ``(asset_id, metric, ts, value)`` with a composite primary key.
Writes use ``INSERT OR REPLACE`` so re-delivered messages (at-least-once
delivery) are idempotent.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS telemetry (
    asset_id TEXT NOT NULL,
    metric   TEXT NOT NULL,
    ts       REAL NOT NULL,
    value    REAL NOT NULL,
    PRIMARY KEY (asset_id, metric, ts)
) WITHOUT ROWID;
"""


class TimeSeriesStore:
    """Thread-safe sqlite time-series store."""

    def __init__(self, path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)

    def write(self, asset_id: str, ts: float, metrics: dict[str, float]) -> int:
        """Write one sample (several metrics at one timestamp). Returns rows written."""
        rows = [(asset_id, name, ts, float(value)) for name, value in metrics.items()]
        return self.write_rows(rows)

    def write_rows(self, rows: Iterable[tuple[str, str, float, float]]) -> int:
        """Bulk write ``(asset_id, metric, ts, value)`` rows."""
        rows = list(rows)
        with self._lock:
            self._conn.executemany("INSERT OR REPLACE INTO telemetry VALUES (?, ?, ?, ?)", rows)
            self._conn.commit()
        return len(rows)

    def query(
        self,
        asset_id: str,
        metric: str,
        start: float | None = None,
        end: float | None = None,
        limit: int | None = None,
    ) -> list[tuple[float, float]]:
        """Return ``(ts, value)`` pairs in ascending time order.

        ``start`` is inclusive and ``end`` exclusive. With ``limit`` the most
        recent ``limit`` points inside the window are returned.
        """
        sql = "SELECT ts, value FROM telemetry WHERE asset_id = ? AND metric = ?"
        args: list[object] = [asset_id, metric]
        if start is not None:
            sql += " AND ts >= ?"
            args.append(start)
        if end is not None:
            sql += " AND ts < ?"
            args.append(end)
        sql += " ORDER BY ts DESC"
        if limit is not None:
            sql += " LIMIT ?"
            args.append(int(limit))
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return rows[::-1]

    def series(self, asset_id: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
        """Full series of a metric as ``(ts, values)`` numpy arrays."""
        rows = self.query(asset_id, metric)
        if not rows:
            return np.empty(0), np.empty(0)
        arr = np.asarray(rows, dtype=float)
        return arr[:, 0], arr[:, 1]

    def latest(self, asset_id: str) -> dict[str, tuple[float, float]]:
        """Latest ``(ts, value)`` of every metric of an asset."""
        sql = (
            "SELECT metric, ts, value FROM telemetry t WHERE asset_id = ? AND ts = "
            "(SELECT MAX(ts) FROM telemetry WHERE asset_id = t.asset_id AND metric = t.metric)"
        )
        with self._lock:
            rows = self._conn.execute(sql, [asset_id]).fetchall()
        return {metric: (ts, value) for metric, ts, value in rows}

    def metrics(self, asset_id: str) -> list[str]:
        """Names of the metrics stored for an asset."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT metric FROM telemetry WHERE asset_id = ? ORDER BY metric",
                [asset_id],
            ).fetchall()
        return [r[0] for r in rows]

    def count(self) -> int:
        """Total number of stored points."""
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM telemetry").fetchone()[0])

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._conn.close()
