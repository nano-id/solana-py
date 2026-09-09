"""Fee model for delta-neutral funding harvest (screen, not live entry)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FeeConfig(BaseModel):
    spot_taker: float = Field(default=0.001, ge=0, le=0.01)
    perp_taker: float = Field(default=0.0005, ge=0, le=0.01)
    use_taker: bool = True
    spot_maker: float = Field(default=0.0008, ge=0, le=0.01)
    perp_maker: float = Field(default=0.0002, ge=0, le=0.01)


def leg_fee(config: FeeConfig, *, venue: str) -> float:
    if config.use_taker:
        return config.spot_taker if venue == "spot" else config.perp_taker
    return config.spot_maker if venue == "spot" else config.perp_maker


def round_trip_cost(config: FeeConfig) -> float:
    """Open + close both spot and perp legs (4 fee events)."""
    return 2.0 * (leg_fee(config, venue="spot") + leg_fee(config, venue="perp"))


def fee_drag_annualized(config: FeeConfig, *, hold_days: float) -> float:
    if hold_days <= 0:
        return float("inf")
    return round_trip_cost(config) * (365.0 / float(hold_days))
