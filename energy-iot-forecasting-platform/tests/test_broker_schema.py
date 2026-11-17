import pytest
from pydantic import ValidationError

from energy_iot.broker import InProcessBroker, topic_matches
from energy_iot.schemas import Telemetry, parse_topic, telemetry_topic


@pytest.mark.parametrize(
    ("pattern", "topic", "expected"),
    [
        ("site/a/pv/p1/telemetry", "site/a/pv/p1/telemetry", True),
        ("site/+/pv/+/telemetry", "site/a/pv/p1/telemetry", True),
        ("site/+/pv/+/telemetry", "site/a/wind/w1/telemetry", False),
        ("site/#", "site/a/pv/p1/telemetry", True),
        ("site/a/#", "site/b/pv/p1/telemetry", False),
        ("site/+", "site/a/pv", False),
        ("site/a/pv/p1/telemetry/extra", "site/a/pv/p1/telemetry", False),
    ],
)
def test_topic_matching(pattern, topic, expected):
    assert topic_matches(pattern, topic) is expected


def test_broker_delivers_to_matching_subscribers_only():
    broker = InProcessBroker()
    got_pv, got_all = [], []
    broker.subscribe("site/+/pv/+/telemetry", got_pv.append)
    broker.subscribe("site/#", got_all.append)
    assert broker.publish("site/a/pv/p1/telemetry", {"x": 1}) == 2
    assert broker.publish("site/a/load/l1/telemetry", {"x": 2}) == 1
    assert len(got_pv) == 1 and len(got_all) == 2
    assert broker.stats.published == 2 and broker.stats.delivered == 3


def test_broker_isolates_failing_handler():
    broker = InProcessBroker()
    received = []

    def boom(_msg):
        raise RuntimeError("consumer bug")

    broker.subscribe("t/#", boom)
    broker.subscribe("t/#", received.append)
    assert broker.publish("t/x", {"v": 1}) == 1
    assert len(received) == 1
    assert broker.stats.handler_errors == 1
    assert "consumer bug" in broker.dead_letters[0][1]


def test_broker_unsubscribe_and_invalid_topics():
    broker = InProcessBroker()
    sub = broker.subscribe("a/b", lambda m: None)
    assert broker.unsubscribe(sub) is True
    assert broker.unsubscribe(sub) is False
    assert broker.publish("a/b", {}) == 0
    assert broker.stats.unrouted == 1
    with pytest.raises(ValueError):
        broker.publish("a/+/b", {})
    with pytest.raises(ValueError):
        broker.subscribe("a/#/b", lambda m: None)


def _pv_payload(**overrides):
    payload = {
        "site": "s1",
        "asset_id": "pv-1",
        "asset_type": "pv",
        "ts": 1_780_000_000.0,
        "metrics": {
            "irradiance_wm2": 500.0,
            "cell_temp_c": 30.0,
            "voltage_v": 180.0,
            "current_a": 8.0,
            "power_w": 1440.0,
        },
    }
    payload.update(overrides)
    return payload


def test_telemetry_valid_and_topic_roundtrip():
    sample = Telemetry.model_validate(_pv_payload())
    assert sample.source == "simulated"
    assert sample.topic() == telemetry_topic("s1", "pv", "pv-1")
    assert parse_topic(sample.topic()) == ("s1", "pv", "pv-1")


@pytest.mark.parametrize(
    "overrides",
    [
        {"metrics": {"power_w": 1.0}},
        {"asset_type": "nuclear"},
        {"ts": -5.0},
        {"unexpected": 1},
        {"schema_version": "9.9"},
        {"asset_id": "bad/id"},
    ],
)
def test_telemetry_rejects_invalid(overrides):
    with pytest.raises(ValidationError):
        Telemetry.model_validate(_pv_payload(**overrides))


def test_telemetry_rejects_out_of_range_and_nan():
    bad = _pv_payload()
    bad["metrics"] = {**bad["metrics"], "irradiance_wm2": 5000.0}
    with pytest.raises(ValidationError):
        Telemetry.model_validate(bad)
    nan = _pv_payload()
    nan["metrics"] = {**nan["metrics"], "power_w": float("nan")}
    with pytest.raises(ValidationError):
        Telemetry.model_validate(nan)
