import pytest
from fastapi.testclient import TestClient

from energy_iot.api import create_app
from energy_iot.platform import EnergyPlatform, SiteConfig


@pytest.fixture(scope="module")
def client(platform):
    return TestClient(create_app(platform))


def test_health_and_assets(client):
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["data_source"] == "simulated"
    assert health["assets"] == 5 and health["messages_rejected"] == 0
    assets = client.get("/assets").json()
    assert {a["asset_id"] for a in assets} == {
        "demo-site",
        "pv-array-1",
        "wind-1",
        "battery-1",
        "load-1",
    }
    assert [a["asset_id"] for a in client.get("/assets", params={"asset_type": "pv"}).json()] == [
        "pv-array-1"
    ]
    site = client.get("/assets/demo-site").json()
    assert len(site["children"]) == 4
    assert client.get("/assets/nope").status_code == 404


def test_telemetry_query(client):
    body = client.get("/assets/battery-1/telemetry", params={"metric": "soc", "limit": 5}).json()
    assert len(body["points"]) == 5
    ts = [p["ts"] for p in body["points"]]
    assert ts == sorted(ts)
    assert all(0.0 <= p["value"] <= 1.0 for p in body["points"])
    assert client.get("/assets/battery-1/telemetry", params={"metric": "x"}).status_code == 404
    assert client.get("/assets/load-1/telemetry", params={"limit": 0}).status_code == 422


def test_forecast_endpoint(client):
    body = client.get("/forecasts/load", params={"horizon_steps": 2}).json()
    assert [p["lead_steps"] for p in body["forecast"]] == [1, 2]
    assert body["forecast"][0]["ts"] > 0 and body["forecast"][0]["power_w"] > 0
    assert body["backtest"]["rmse_model"] > 0
    assert body["data_source"] == "simulated"
    assert client.get("/forecasts/coal").status_code == 404


def test_forecast_needs_history_and_dispatch_needs_steps():
    empty = TestClient(create_app(EnergyPlatform(SiteConfig())))
    assert empty.get("/forecasts/pv").status_code == 409
    assert empty.get("/dispatch/summary").status_code == 409
    assert empty.get("/anomalies/pv").json()["count"] == 0


def test_post_telemetry_and_dispatch_summary():
    p = EnergyPlatform(SiteConfig())
    p.run(hours=24)
    c = TestClient(create_app(p))
    sample = {
        "site": "demo-site",
        "asset_id": "load-1",
        "asset_type": "load",
        "ts": p.now + 900.0,
        "metrics": {"power_w": 432.0},
    }
    assert c.post("/telemetry", json=sample).status_code == 202
    assert c.get("/assets/load-1").json()["state"]["reported"]["power_w"] == 432.0
    assert c.post("/telemetry", json={**sample, "asset_id": "ghost"}).status_code == 422
    assert c.post("/telemetry", json={**sample, "metrics": {}}).status_code == 422
    summary = c.get("/dispatch/summary").json()
    assert summary["strategy"] == "self_consumption"
    assert summary["load_kwh"] > 0 and "pv" in summary["site_power_w"]
    anomalies = c.get("/anomalies/pv").json()
    assert anomalies["count"] == len(anomalies["anomalies"])
