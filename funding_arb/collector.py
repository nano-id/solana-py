"""Live snapshot collector: funding rate + spot/perp prices + order-book depth."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from funding_arb.config import DepthConfig, FundingArbAppConfig, load_funding_arb_config
from funding_arb.depth import fetch_order_book_safe, summarize_order_book
from pairs_trading.data.exchange import ConnectedExchange, connect_with_fallback
from pairs_trading.funding.basis import fetch_basis_snapshot
from pairs_trading.funding.symbols import swap_to_spot

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def collect_symbol_snapshot(
    exchange: Any,
    swap_symbol: str,
    *,
    depth: DepthConfig,
    exchange_id: str,
    observed_at: str | None = None,
) -> dict[str, Any]:
    """One-row live snapshot for a swap symbol (reuses basis fetch + depth)."""
    observed_at = observed_at or _utc_now_iso()
    basis = fetch_basis_snapshot(exchange, swap_symbol)
    spot_symbol = str(basis.get("spot_symbol") or swap_to_spot(swap_symbol))

    perp_depth: dict[str, float | None] = {
        "mid": None,
        "best_bid": None,
        "best_ask": None,
        "spread_bps": None,
        "bid_notional": None,
        "ask_notional": None,
        "min_notional": None,
    }
    spot_depth = dict(perp_depth)

    if depth.fetch_perp_book:
        book = fetch_order_book_safe(exchange, swap_symbol, limit=depth.book_limit)
        perp_depth = summarize_order_book(book, band_frac=depth.band_frac)
    if depth.fetch_spot_book:
        book = fetch_order_book_safe(exchange, spot_symbol, limit=depth.book_limit)
        spot_depth = summarize_order_book(book, band_frac=depth.band_frac)

    # Conservative depth for delta-neutral: limited by thinner of the two books.
    depth_usdt = None
    p_min = perp_depth.get("min_notional")
    s_min = spot_depth.get("min_notional")
    if p_min is not None and s_min is not None:
        depth_usdt = float(min(p_min, s_min))
    elif p_min is not None:
        depth_usdt = float(p_min)
    elif s_min is not None:
        depth_usdt = float(s_min)

    return {
        "observed_at": observed_at,
        "exchange_id": exchange_id,
        "symbol": swap_symbol,
        "spot_symbol": spot_symbol,
        "funding_rate": basis.get("funding_rate"),
        "mark_price": basis.get("mark_price"),
        "index_price": basis.get("index_price"),
        "perp_last": basis.get("mark_price"),
        "spot_last": basis.get("spot_last"),
        "quote_volume_24h": basis.get("quote_volume_24h"),
        "basis_bps_spot": basis.get("basis_bps_spot"),
        "perp_spread_bps": perp_depth.get("spread_bps"),
        "perp_bid_notional": perp_depth.get("bid_notional"),
        "perp_ask_notional": perp_depth.get("ask_notional"),
        "perp_min_notional": perp_depth.get("min_notional"),
        "spot_spread_bps": spot_depth.get("spread_bps"),
        "spot_bid_notional": spot_depth.get("bid_notional"),
        "spot_ask_notional": spot_depth.get("ask_notional"),
        "spot_min_notional": spot_depth.get("min_notional"),
        "depth_usdt": depth_usdt,
        "depth_band_frac": depth.band_frac,
    }


def collect_universe_snapshot(
    exchange: Any,
    symbols: list[str],
    *,
    depth: DepthConfig,
    exchange_id: str,
) -> pd.DataFrame:
    observed_at = _utc_now_iso()
    rows = [
        collect_symbol_snapshot(
            exchange,
            sym,
            depth=depth,
            exchange_id=exchange_id,
            observed_at=observed_at,
        )
        for sym in symbols
    ]
    return pd.DataFrame(rows)


def connect_collector(config: FundingArbAppConfig | None = None) -> ConnectedExchange:
    config = config or load_funding_arb_config()
    return connect_with_fallback(config.exchange)


def collect_once(
    config: FundingArbAppConfig | None = None,
    *,
    symbols: list[str] | None = None,
    connected: ConnectedExchange | None = None,
) -> tuple[pd.DataFrame, ConnectedExchange]:
    """Fetch one full-universe snapshot; optionally reuse an open connection."""
    config = config or load_funding_arb_config()
    connected = connected or connect_collector(config)
    syms = symbols or list(config.collector.symbols)
    frame = collect_universe_snapshot(
        connected.exchange,
        syms,
        depth=config.depth,
        exchange_id=connected.exchange_id,
    )
    return frame, connected
