"""Capacity / liquidity filters for funding harvest candidates."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, Field


class CapacityConfig(BaseModel):
    min_quote_volume_24h: float = Field(default=5_000_000.0, ge=0)
    max_adv_fraction: float = Field(default=0.01, gt=0, le=0.5)
    max_notional_usdt: float = Field(default=50_000.0, gt=0)


def passes_capacity(quote_volume_24h: float | None, config: CapacityConfig) -> bool:
    if quote_volume_24h is None or quote_volume_24h != quote_volume_24h:
        return False
    return float(quote_volume_24h) >= config.min_quote_volume_24h


def max_position_notional(quote_volume_24h: float | None, config: CapacityConfig) -> float:
    if quote_volume_24h is None or quote_volume_24h != quote_volume_24h:
        return 0.0
    return float(min(float(quote_volume_24h) * config.max_adv_fraction, config.max_notional_usdt))


def apply_capacity_filter(df: pd.DataFrame, config: CapacityConfig) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "quote_volume_24h" not in out.columns:
        out["passes_capacity"] = False
        out["max_notional"] = 0.0
        return out
    out["passes_capacity"] = out["quote_volume_24h"].map(lambda v: passes_capacity(v, config))
    out["max_notional"] = out["quote_volume_24h"].map(lambda v: max_position_notional(v, config))
    return out


def enrich_quote_volumes(exchange: Any, symbols: list[str]) -> dict[str, float | None]:
    from pairs_trading.funding.basis import _quote_volume_24h

    out: dict[str, float | None] = {}
    for sym in symbols:
        try:
            tick = exchange.fetch_ticker(sym)
            out[sym] = _quote_volume_24h(tick)
        except Exception:
            out[sym] = None
    return out
