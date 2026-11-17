"""Digital-twin registry: asset models, relationships and last reported state.

The registry is a small in-memory analogue of a twin graph. Each asset has
static properties (the model), an optional parent (``contained_in``
relationship) and a state that is updated by the ingestion service.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .schemas import AssetType

Status = Literal["unknown", "online", "stale"]


class Asset(BaseModel):
    """Static description of an asset."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str = Field(min_length=1)
    asset_type: AssetType | Literal["site"]
    site: str
    name: str = ""
    parent_id: str | None = None
    rated_power_w: float | None = Field(default=None, ge=0)
    properties: dict[str, Any] = Field(default_factory=dict)


@dataclass
class TwinState:
    """Last reported state of an asset."""

    last_seen: float | None = None
    reported: dict[str, float] = field(default_factory=dict)
    message_count: int = 0


class TwinRegistry:
    """Registry of assets and their reported state."""

    def __init__(self, stale_after_s: float = 900.0) -> None:
        self.stale_after_s = stale_after_s
        self._assets: dict[str, Asset] = {}
        self._state: dict[str, TwinState] = {}

    def register(self, asset: Asset) -> Asset:
        """Add an asset. The parent, if given, must already be registered."""
        if asset.asset_id in self._assets:
            raise ValueError(f"asset {asset.asset_id!r} is already registered")
        if asset.parent_id is not None and asset.parent_id not in self._assets:
            raise ValueError(f"unknown parent {asset.parent_id!r}")
        self._assets[asset.asset_id] = asset
        self._state[asset.asset_id] = TwinState()
        return asset

    def __contains__(self, asset_id: str) -> bool:
        return asset_id in self._assets

    def get(self, asset_id: str) -> Asset:
        """Return an asset or raise ``KeyError``."""
        return self._assets[asset_id]

    def list(self, asset_type: str | None = None) -> list[Asset]:
        """All assets, optionally filtered by type, ordered by id."""
        assets = sorted(self._assets.values(), key=lambda a: a.asset_id)
        if asset_type is None:
            return assets
        return [a for a in assets if a.asset_type == asset_type]

    def children(self, asset_id: str) -> list[Asset]:
        """Assets whose parent is ``asset_id``."""
        return [a for a in self.list() if a.parent_id == asset_id]

    def update_state(self, asset_id: str, ts: float, metrics: dict[str, float]) -> bool:
        """Merge reported metrics. Samples older than the last one are ignored.

        Returns True if the state was updated.
        """
        state = self._state[asset_id]
        state.message_count += 1
        if state.last_seen is not None and ts < state.last_seen:
            return False
        state.last_seen = ts
        state.reported.update(metrics)
        return True

    def state(self, asset_id: str) -> TwinState:
        """Reported state of an asset."""
        return self._state[asset_id]

    def status(self, asset_id: str, now: float) -> Status:
        """Connectivity status derived from the age of the last sample."""
        last = self._state[asset_id].last_seen
        if last is None:
            return "unknown"
        return "online" if now - last <= self.stale_after_s else "stale"

    def describe(self, asset_id: str, now: float | None = None) -> dict[str, Any]:
        """Twin document: model, relationships and state."""
        asset = self.get(asset_id)
        state = self._state[asset_id]
        ref_time = now if now is not None else (state.last_seen or 0.0)
        return {
            **asset.model_dump(),
            "children": [c.asset_id for c in self.children(asset_id)],
            "state": {
                "status": self.status(asset_id, ref_time),
                "last_seen": state.last_seen,
                "reported": dict(state.reported),
                "message_count": state.message_count,
            },
        }

    def site_power_balance(self, site_id: str) -> dict[str, float]:
        """Aggregate last reported power of the children of a site by asset type."""
        totals: dict[str, float] = {}
        for child in self.children(site_id):
            power = self._state[child.asset_id].reported.get("power_w")
            if power is not None:
                totals[child.asset_type] = totals.get(child.asset_type, 0.0) + power
        return totals
