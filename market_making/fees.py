"""Fee helpers and positivity gate."""

from __future__ import annotations

from market_making.config import MarketMakingConfig


def fee_on_notional(notional: float, cfg: MarketMakingConfig) -> float:
    return abs(notional) * cfg.commission_frac


def round_trip_net_edge_bps(half_spread_bps: float, cfg: MarketMakingConfig) -> float:
    """Net bps if bid and ask both fill at quote (no adverse selection)."""
    return 2.0 * half_spread_bps - 2.0 * cfg.commission_bps


def is_positive_half_spread(half_spread_bps: float, cfg: MarketMakingConfig) -> bool:
    return (
        half_spread_bps + 1e-12 >= cfg.required_half_spread_bps
        and round_trip_net_edge_bps(half_spread_bps, cfg) > 0.0
    )
