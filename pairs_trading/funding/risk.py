"""Delta-neutral position sizing for funding harvest."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from pairs_trading.funding.capacity import CapacityConfig, max_position_notional


class RiskConfig(BaseModel):
    equity_usdt: float = Field(default=10_000.0, gt=0)
    max_name_fraction: float = Field(default=0.25, gt=0, le=1.0)
    target_gross_fraction: float = Field(default=0.5, gt=0, le=5.0)
    min_notional_usdt: float = Field(default=50.0, ge=0)


@dataclass(frozen=True)
class SizedPosition:
    symbol: str
    side: str
    notional_usdt: float
    equity_usdt: float
    reason: str


def harvest_side(mean_or_last_rate: float) -> str:
    if mean_or_last_rate >= 0:
        return "short_perp_long_spot"
    return "long_perp_short_spot"


def size_position(
    symbol: str,
    *,
    funding_rate_signal: float,
    quote_volume_24h: float | None,
    risk: RiskConfig,
    capacity: CapacityConfig,
) -> SizedPosition:
    side = harvest_side(funding_rate_signal)
    by_equity = risk.equity_usdt * risk.target_gross_fraction
    by_name = risk.equity_usdt * risk.max_name_fraction
    uncapped = min(by_equity, by_name)
    by_cap = max_position_notional(quote_volume_24h, capacity)
    # Unknown volume → still size for paper/diagnostics; tradable gate uses capacity.
    if by_cap <= 0:
        if uncapped < risk.min_notional_usdt:
            return SizedPosition(symbol, side, 0.0, risk.equity_usdt, "below_min_notional")
        return SizedPosition(symbol, side, float(uncapped), risk.equity_usdt, "ok_uncapped_volume")
    notional = min(uncapped, by_cap)
    if notional < risk.min_notional_usdt:
        return SizedPosition(symbol, side, 0.0, risk.equity_usdt, "below_min_notional")
    return SizedPosition(symbol, side, float(notional), risk.equity_usdt, "ok")
