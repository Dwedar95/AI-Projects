"""In-process publish/subscribe bus with MQTT-style topics.

Topics are slash-separated strings. Subscriptions may use the MQTT wildcards ``+``
(exactly one level) and ``#`` (all remaining levels, only as the last level). Delivery
is synchronous and in subscription order, which keeps simulations deterministic.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

Handler = Callable[["Message"], None]


@dataclass(frozen=True)
class Message:
    seq: int
    topic: str
    payload: dict[str, Any]
    retained: bool = False


def topic_matches(pattern: str, topic: str) -> bool:
    """MQTT topic filter matching."""
    p_levels, t_levels = pattern.split("/"), topic.split("/")
    for i, p in enumerate(p_levels):
        if p == "#":
            return i == len(p_levels) - 1
        if i >= len(t_levels):
            return False
        if p != "+" and p != t_levels[i]:
            return False
    return len(p_levels) == len(t_levels)


class MessageBus:
    """Minimal broker: subscribe, publish, retained messages and a message log."""

    def __init__(self, keep_log: bool = True) -> None:
        self._subs: list[tuple[str, Handler]] = []
        self._retained: dict[str, Message] = {}
        self._seq = 0
        self.keep_log = keep_log
        self.log: list[Message] = []
        self.counts: Counter[str] = Counter()

    def subscribe(self, pattern: str, handler: Handler) -> None:
        """Register a handler; retained messages that match are delivered at once."""
        if "#" in pattern.split("/")[:-1]:
            raise ValueError("'#' is only allowed as the last topic level")
        self._subs.append((pattern, handler))
        for topic, msg in self._retained.items():
            if topic_matches(pattern, topic):
                handler(msg)

    def publish(self, topic: str, payload: dict[str, Any], retain: bool = False) -> Message:
        """Deliver a message to every matching subscriber and return it."""
        if "+" in topic or "#" in topic:
            raise ValueError("wildcards are not allowed in published topics")
        self._seq += 1
        msg = Message(self._seq, topic, payload, retain)
        if retain:
            self._retained[topic] = msg
        if self.keep_log:
            self.log.append(msg)
        self.counts[topic] += 1
        for pattern, handler in list(self._subs):
            if topic_matches(pattern, topic):
                handler(msg)
        return msg

    def count(self, pattern: str) -> int:
        """Number of messages published so far on topics matching ``pattern``."""
        return sum(n for topic, n in self.counts.items() if topic_matches(pattern, topic))
