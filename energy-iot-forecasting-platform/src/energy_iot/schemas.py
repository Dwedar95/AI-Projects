"""Telemetry message schema and topic convention.

Topic convention: ``site/<site>/<asset_type>/<asset_id>/telemetry``.
Every message carries a ``source`` field. All data produced by this
repository is ``"simulated"``.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0"

AssetType = Literal["pv", "wind", "battery", "load"]

REQUIRED_METRICS: dict[str, tuple[str, ...]] = {
    "pv": ("irradiance_wm2", "cell_temp_c", "voltage_v", "current_a", "power_w"),
    "wind": ("wind_speed_ms", "power_w"),
    "battery": ("soc", "power_w"),
    "load": ("power_w",),
}

# Physical plausibility limits (inclusive). Values outside are rejected.
METRIC_BOUNDS: dict[str, tuple[float, float]] = {
    "irradiance_wm2": (0.0, 1500.0),
    "cell_temp_c": (-40.0, 120.0),
    "wind_speed_ms": (0.0, 60.0),
    "soc": (0.0, 1.0),
    "voltage_v": (0.0, 2000.0),
}


class Telemetry(BaseModel):
    """One telemetry sample of one asset."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    site: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    asset_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    asset_type: AssetType
    ts: float = Field(gt=0, description="Unix timestamp in seconds (UTC)")
    metrics: dict[str, float]
    source: Literal["simulated", "field"] = "simulated"

    @model_validator(mode="after")
    def _check_metrics(self) -> Telemetry:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {self.schema_version!r}")
        missing = [m for m in REQUIRED_METRICS[self.asset_type] if m not in self.metrics]
        if missing:
            raise ValueError(f"missing metrics for {self.asset_type}: {missing}")
        for name, value in self.metrics.items():
            if not math.isfinite(value):
                raise ValueError(f"metric {name} is not finite")
            bounds = METRIC_BOUNDS.get(name)
            if bounds and not bounds[0] <= value <= bounds[1]:
                raise ValueError(f"metric {name}={value} outside {bounds}")
        return self

    def topic(self) -> str:
        """Topic this sample is published on."""
        return telemetry_topic(self.site, self.asset_type, self.asset_id)


def telemetry_topic(site: str, asset_type: str, asset_id: str) -> str:
    """Build the telemetry topic of an asset."""
    return f"site/{site}/{asset_type}/{asset_id}/telemetry"


def parse_topic(topic: str) -> tuple[str, str, str]:
    """Split a telemetry topic into ``(site, asset_type, asset_id)``."""
    parts = topic.split("/")
    if len(parts) != 5 or parts[0] != "site" or parts[4] != "telemetry":
        raise ValueError(f"not a telemetry topic: {topic!r}")
    return parts[1], parts[2], parts[3]
