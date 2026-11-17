import numpy as np
import pytest

from energy_iot.platform import EnergyPlatform, SiteConfig
from energy_iot.store import TimeSeriesStore
from energy_iot.twin import Asset, TwinRegistry


def test_store_write_query_and_idempotency():
    store = TimeSeriesStore()
    for k in range(10):
        store.write("a1", 100.0 + k, {"power_w": float(k), "soc": 0.5})
    store.write("a1", 105.0, {"power_w": 55.0})  # redelivery overwrites
    assert store.count() == 20
    rows = store.query("a1", "power_w", start=103.0, end=107.0)
    assert rows == [(103.0, 3.0), (104.0, 4.0), (105.0, 55.0), (106.0, 6.0)]
    assert store.query("a1", "power_w", limit=2) == [(108.0, 8.0), (109.0, 9.0)]
    assert store.latest("a1") == {"power_w": (109.0, 9.0), "soc": (109.0, 0.5)}
    assert store.metrics("a1") == ["power_w", "soc"]
    ts, values = store.series("missing", "power_w")
    assert len(ts) == 0 and len(values) == 0


def test_twin_registry_relationships_and_status():
    reg = TwinRegistry(stale_after_s=60.0)
    reg.register(Asset(asset_id="site", asset_type="site", site="site"))
    reg.register(Asset(asset_id="pv", asset_type="pv", site="site", parent_id="site"))
    with pytest.raises(ValueError):
        reg.register(Asset(asset_id="pv", asset_type="pv", site="site"))
    with pytest.raises(ValueError):
        reg.register(Asset(asset_id="x", asset_type="pv", site="site", parent_id="nope"))
    assert [a.asset_id for a in reg.children("site")] == ["pv"]
    assert reg.status("pv", now=0.0) == "unknown"
    assert reg.update_state("pv", 1000.0, {"power_w": 5.0}) is True
    assert reg.update_state("pv", 900.0, {"power_w": 1.0}) is False  # out of order
    assert reg.state("pv").reported["power_w"] == 5.0
    assert reg.status("pv", now=1030.0) == "online"
    assert reg.status("pv", now=2000.0) == "stale"
    assert reg.site_power_balance("site") == {"pv": 5.0}


def test_platform_ingests_all_simulated_messages(platform):
    steps = 96 * 4
    assert platform.ingestion.stats.accepted == 4 * steps
    assert platform.ingestion.stats.rejected == 0
    # pv 5 metrics, wind 2, battery 2, load 1
    assert platform.store.count() == 10 * steps
    twin = platform.registry.describe(platform.PV_ID, platform.now)
    assert twin["state"]["status"] == "online"
    assert twin["state"]["message_count"] == steps
    assert twin["parent_id"] == "demo-site"


def test_ingestion_rejects_bad_messages():
    p = EnergyPlatform(SiteConfig())
    good = {
        "site": "demo-site",
        "asset_id": "load-1",
        "asset_type": "load",
        "ts": 2_000_000_000.0,
        "metrics": {"power_w": 100.0},
    }
    p.broker.publish("site/demo-site/load/load-1/telemetry", good)
    p.broker.publish("site/demo-site/load/load-1/telemetry", {**good, "metrics": {}})
    p.broker.publish("site/other/load/load-1/telemetry", good)
    p.broker.publish("site/demo-site/load/ghost/telemetry", {**good, "asset_id": "ghost"})
    p.broker.publish(
        "site/demo-site/wind/load-1/telemetry",
        {**good, "asset_type": "wind", "metrics": {"wind_speed_ms": 5.0, "power_w": 1.0}},
    )
    assert p.ingestion.stats.accepted == 1
    assert p.ingestion.stats.reasons == {
        "schema": 1,
        "topic_mismatch": 1,
        "unknown_asset": 1,
        "type_mismatch": 1,
    }
    assert p.store.count() == 1


def test_run_is_reproducible_and_independent_of_call_splitting():
    a = EnergyPlatform(SiteConfig(seed=11))
    a.run(hours=24)
    b = EnergyPlatform(SiteConfig(seed=11))
    b.run(hours=6)
    b.run(hours=18)
    for asset, metric in [("pv-array-1", "power_w"), ("battery-1", "soc"), ("load-1", "power_w")]:
        assert np.array_equal(a.store.series(asset, metric)[1], b.store.series(asset, metric)[1])


def test_site_energy_balance_closes(platform):
    pv = platform.store.series(platform.PV_ID, "power_w")[1]
    wind = platform.store.series(platform.WIND_ID, "power_w")[1]
    load = platform.store.series(platform.LOAD_ID, "power_w")[1]
    batt = platform.store.series(platform.BATTERY_ID, "power_w")[1]
    res = platform.dispatch_summary()
    residual = pv + wind + res.grid_import_w - load - batt - res.grid_export_w
    assert np.max(np.abs(residual)) < 1e-6
    assert 0.0 <= res.self_sufficiency <= 1.0 and 0.0 <= res.self_consumption <= 1.0
