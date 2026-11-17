"""MQTT-style publish/subscribe abstraction.

The platform talks to a :class:`Broker` protocol. The default implementation is
:class:`InProcessBroker`, which delivers messages synchronously inside the
Python process and supports MQTT topic wildcards (``+`` for one level, ``#``
for the remaining levels). :class:`PahoMqttBroker` is an optional adapter for a
real MQTT broker and needs the ``paho-mqtt`` package.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

Handler = Callable[["Message"], None]


@dataclass(frozen=True)
class Message:
    """A published message: topic plus a JSON-serialisable payload."""

    topic: str
    payload: dict[str, Any]


def validate_topic(topic: str) -> None:
    """Raise ``ValueError`` if ``topic`` is not a valid publish topic."""
    if not topic or topic.startswith("/") or topic.endswith("/"):
        raise ValueError(f"invalid topic: {topic!r}")
    if "+" in topic or "#" in topic:
        raise ValueError(f"wildcards are not allowed in a publish topic: {topic!r}")


def topic_matches(pattern: str, topic: str) -> bool:
    """Return True if ``topic`` matches the MQTT subscription ``pattern``.

    ``+`` matches exactly one level, ``#`` matches all remaining levels and
    must be the last level of the pattern.
    """
    p_levels = pattern.split("/")
    t_levels = topic.split("/")
    for i, level in enumerate(p_levels):
        if level == "#":
            return i == len(p_levels) - 1
        if i >= len(t_levels):
            return False
        if level != "+" and level != t_levels[i]:
            return False
    return len(p_levels) == len(t_levels)


class Broker(Protocol):
    """Minimal broker interface used by simulators and services."""

    def publish(self, topic: str, payload: dict[str, Any]) -> int:
        """Publish a payload and return the number of handlers it reached."""
        ...

    def subscribe(self, pattern: str, handler: Handler) -> int:
        """Register a handler for a topic pattern and return a subscription id."""
        ...


@dataclass
class BrokerStats:
    """Counters kept by the in-process broker."""

    published: int = 0
    delivered: int = 0
    unrouted: int = 0
    handler_errors: int = 0


@dataclass
class InProcessBroker:
    """Synchronous in-process broker with MQTT wildcard semantics.

    A failing handler does not stop delivery to the other subscribers. The
    failure is counted and the message is kept in ``dead_letters`` together
    with the error text, so one bad consumer cannot block the telemetry path.
    """

    stats: BrokerStats = field(default_factory=BrokerStats)
    dead_letters: list[tuple[Message, str]] = field(default_factory=list)
    _subs: dict[int, tuple[str, Handler]] = field(default_factory=dict)
    _next_id: int = 1

    def subscribe(self, pattern: str, handler: Handler) -> int:
        if not pattern:
            raise ValueError("empty subscription pattern")
        if "#" in pattern and not pattern.endswith("#"):
            raise ValueError("'#' is only allowed as the last topic level")
        sub_id = self._next_id
        self._next_id += 1
        self._subs[sub_id] = (pattern, handler)
        return sub_id

    def unsubscribe(self, sub_id: int) -> bool:
        """Remove a subscription. Returns False if the id is unknown."""
        return self._subs.pop(sub_id, None) is not None

    def publish(self, topic: str, payload: dict[str, Any]) -> int:
        validate_topic(topic)
        message = Message(topic=topic, payload=payload)
        self.stats.published += 1
        reached = 0
        for pattern, handler in list(self._subs.values()):
            if not topic_matches(pattern, topic):
                continue
            try:
                handler(message)
                reached += 1
            except Exception as exc:  # noqa: BLE001 - isolate consumers from each other
                self.stats.handler_errors += 1
                self.dead_letters.append((message, f"{type(exc).__name__}: {exc}"))
        if reached == 0:
            self.stats.unrouted += 1
        self.stats.delivered += reached
        return reached


class PahoMqttBroker:
    """Optional adapter that maps the :class:`Broker` protocol to paho-mqtt.

    Not exercised by the test suite because it needs a reachable MQTT broker.
    Install with ``pip install paho-mqtt`` (extra ``mqtt``). Payloads are
    encoded as JSON. ``publish`` returns 0 because the number of remote
    receivers is unknown to an MQTT client.
    """

    def __init__(self, host: str, port: int = 1883, client_id: str = "energy-iot") -> None:
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("paho-mqtt is not installed: pip install paho-mqtt") from exc
        self._handlers: dict[int, tuple[str, Handler]] = {}
        self._next_id = 1
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        self._client.on_message = self._on_message
        self._client.connect(host, port)
        self._client.loop_start()

    def _on_message(self, _client: Any, _userdata: Any, msg: Any) -> None:
        payload = json.loads(msg.payload.decode("utf-8"))
        message = Message(topic=msg.topic, payload=payload)
        for pattern, handler in list(self._handlers.values()):
            if topic_matches(pattern, msg.topic):
                handler(message)

    def publish(self, topic: str, payload: dict[str, Any]) -> int:
        validate_topic(topic)
        self._client.publish(topic, json.dumps(payload), qos=1)
        return 0

    def subscribe(self, pattern: str, handler: Handler) -> int:
        sub_id = self._next_id
        self._next_id += 1
        self._handlers[sub_id] = (pattern, handler)
        self._client.subscribe(pattern, qos=1)
        return sub_id

    def close(self) -> None:
        """Stop the network loop and disconnect."""
        self._client.loop_stop()
        self._client.disconnect()
