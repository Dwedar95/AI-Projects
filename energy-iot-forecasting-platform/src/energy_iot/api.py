"""FastAPI gateway for assets, telemetry, forecasts, anomalies and dispatch."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query

from . import __version__
from .platform import EnergyPlatform, SiteConfig
from .schemas import Telemetry


def create_app(platform: EnergyPlatform) -> FastAPI:
    """Build the gateway application around a platform instance."""
    app = FastAPI(
        title="Energy IoT gateway",
        version=__version__,
        description="Reference gateway over simulated telemetry. All data is simulated.",
    )

    def _require_asset(asset_id: str) -> None:
        if asset_id not in platform.registry:
            raise HTTPException(status_code=404, detail=f"unknown asset {asset_id!r}")

    @app.get("/health")
    def health() -> dict[str, Any]:
        stats = platform.ingestion.stats
        return {
            "status": "ok",
            "version": __version__,
            "data_source": "simulated",
            "assets": len(platform.registry.list()),
            "points_stored": platform.store.count(),
            "messages_accepted": stats.accepted,
            "messages_rejected": stats.rejected,
        }

    @app.get("/assets")
    def list_assets(asset_type: str | None = None) -> list[dict[str, Any]]:
        return [
            platform.registry.describe(a.asset_id, platform.now)
            for a in platform.registry.list(asset_type)
        ]

    @app.get("/assets/{asset_id}")
    def get_asset(asset_id: str) -> dict[str, Any]:
        _require_asset(asset_id)
        return platform.registry.describe(asset_id, platform.now)

    @app.get("/assets/{asset_id}/telemetry")
    def get_telemetry(
        asset_id: str,
        metric: str = "power_w",
        start: float | None = None,
        end: float | None = None,
        limit: int = Query(default=200, ge=1, le=5000),
    ) -> dict[str, Any]:
        _require_asset(asset_id)
        if metric not in platform.store.metrics(asset_id):
            raise HTTPException(status_code=404, detail=f"no metric {metric!r} for {asset_id!r}")
        rows = platform.store.query(asset_id, metric, start, end, limit)
        return {
            "asset_id": asset_id,
            "metric": metric,
            "data_source": "simulated",
            "points": [{"ts": ts, "value": value} for ts, value in rows],
        }

    @app.post("/telemetry", status_code=202)
    def post_telemetry(sample: Telemetry) -> dict[str, Any]:
        """HTTP ingestion path: the sample is published on the broker like device traffic."""
        before = platform.ingestion.stats.accepted
        platform.broker.publish(sample.topic(), sample.model_dump())
        if platform.ingestion.stats.accepted == before:
            reason = (
                platform.ingestion.rejected[-1][2] if platform.ingestion.rejected else "rejected"
            )
            raise HTTPException(status_code=422, detail=reason)
        return {"accepted": True, "topic": sample.topic()}

    @app.get("/forecasts/{target}")
    def get_forecast(target: str, horizon_steps: int = Query(default=4, ge=1, le=12)) -> dict:
        try:
            report, points = platform.forecast(target, horizon_steps)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=f"not enough history: {exc}") from exc
        return {
            "target": target,
            "data_source": "simulated",
            "step_seconds": platform.config.dt_s,
            "forecast": points,
            "backtest": report.as_dict(),
        }

    @app.get("/anomalies/pv")
    def get_pv_anomalies(threshold: float = Query(default=4.0, gt=0)) -> dict[str, Any]:
        anomalies = platform.pv_anomalies(threshold)
        return {"asset_id": platform.PV_ID, "count": len(anomalies), "anomalies": anomalies}

    @app.get("/dispatch/summary")
    def get_dispatch_summary() -> dict[str, Any]:
        try:
            result = platform.dispatch_summary()
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "strategy": platform.config.dispatch.strategy,
            "site_power_w": platform.registry.site_power_balance(platform.config.site),
            **result.summary(),
        }

    return app


def demo_app() -> FastAPI:
    """App factory for ``uvicorn energy_iot.api:demo_app --factory`` with 5 simulated days."""
    platform = EnergyPlatform(SiteConfig())
    platform.run(hours=120)
    return create_app(platform)
