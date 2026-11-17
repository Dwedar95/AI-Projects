"""Telemetry ingestion: validate, persist and update the digital twin."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from .broker import Broker, Message
from .schemas import Telemetry, parse_topic
from .store import TimeSeriesStore
from .twin import TwinRegistry

TELEMETRY_PATTERN = "site/+/+/+/telemetry"


@dataclass
class IngestionStats:
    """Counters of the ingestion service."""

    accepted: int = 0
    rejected: int = 0
    reasons: dict[str, int] = field(default_factory=dict)


class IngestionService:
    """Subscribes to telemetry topics and writes valid samples to the store.

    A message is rejected when it fails schema validation, when the topic does
    not agree with the payload, or when the asset is not registered in the
    twin registry. Rejected messages are kept (bounded) for inspection.
    """

    def __init__(
        self,
        broker: Broker,
        store: TimeSeriesStore,
        registry: TwinRegistry,
        max_rejected_kept: int = 100,
    ) -> None:
        self.store = store
        self.registry = registry
        self.stats = IngestionStats()
        self.rejected: list[tuple[str, dict[str, Any], str]] = []
        self._max_rejected_kept = max_rejected_kept
        self.subscription_id = broker.subscribe(TELEMETRY_PATTERN, self.handle)

    def _reject(self, message: Message, reason: str, detail: str) -> None:
        self.stats.rejected += 1
        self.stats.reasons[reason] = self.stats.reasons.get(reason, 0) + 1
        if len(self.rejected) < self._max_rejected_kept:
            self.rejected.append((message.topic, message.payload, f"{reason}: {detail}"))

    def handle(self, message: Message) -> None:
        """Process one broker message."""
        try:
            sample = Telemetry.model_validate(message.payload)
        except ValidationError as exc:
            first = exc.errors()[0]
            self._reject(message, "schema", str(first.get("msg", "invalid")))
            return
        try:
            topic_ids = parse_topic(message.topic)
        except ValueError as exc:
            self._reject(message, "topic", str(exc))
            return
        if topic_ids != (sample.site, sample.asset_type, sample.asset_id):
            self._reject(message, "topic_mismatch", "topic does not match payload identity")
            return
        if sample.asset_id not in self.registry:
            self._reject(message, "unknown_asset", sample.asset_id)
            return
        if self.registry.get(sample.asset_id).asset_type != sample.asset_type:
            self._reject(message, "type_mismatch", "asset type differs from the registered twin")
            return
        self.store.write(sample.asset_id, sample.ts, sample.metrics)
        self.registry.update_state(sample.asset_id, sample.ts, sample.metrics)
        self.stats.accepted += 1
