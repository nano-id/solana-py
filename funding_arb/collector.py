"""Live snapshot collector: funding rate + spot/perp prices + order-book depth."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from funding_arb.config import DepthConfig, FundingArbAppConfig, load_funding_arb_config
from funding_arb.depth import (
    contract_size_for,
    fetch_order_book_safe,
    summarize_order_book,
)
from pairs_trading.data.exchange import ConnectedExchange, connect_with_fallback
from pairs_trading.funding.basis import fetch_basis_snapshot
from pairs_trading.funding.symbols import swap_to_spot

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _f(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def extract_funding_fields(exchange: Any, swap_symbol: str) -> dict[str, Any]:
    """Pull fundingRate + OKX sett/interest without rounding.

    OKX often parks ``fundingRate`` at ``interestRate`` (0.0001) when premium≈0.
    Last settled rate (``settFundingRate``) can differ — we keep both.
    """
    out: dict[str, Any] = {
        "funding_rate": None,
        "funding_rate_raw": None,
        "sett_funding_rate": None,
        "sett_funding_rate_raw": None,
        "interest_rate": None,
        "interest_rate_raw": None,
        "at_interest_floor": False,
    }
    try:
        fr = exchange.fetch_funding_rate(swap_symbol)
    except Exception as exc:  # noqa: BLE001
        logger.warning("fetch_funding_rate failed for %s: %s", swap_symbol, exc)
        return out

    info = fr.get("info") if isinstance(fr.get("info"), dict) else {}

    # Prefer venue raw string (no float round-trip) when present.
    raw = info.get("fundingRate")
    if raw is None or raw == "":
        raw = fr.get("fundingRate")
    out["funding_rate_raw"] = None if raw is None else str(raw)
    out["funding_rate"] = _f(raw)

    sett_raw = info.get("settFundingRate")
    out["sett_funding_rate_raw"] = None if sett_raw in (None, "") else str(sett_raw)
    out["sett_funding_rate"] = _f(sett_raw)

    int_raw = info.get("interestRate")
    out["interest_rate_raw"] = None if int_raw in (None, "") else str(int_raw)
    out["interest_rate"] = _f(int_raw)

    rate = out["funding_rate"]
    interest = out["interest_rate"]
    if rate is not None and interest is not None and abs(rate - interest) < 1e-15:
        out["at_interest_floor"] = True

    # High-precision log for audit (answers "is 0.0001 collector rounding?").
    logger.info(
        "funding raw %s fundingRate=%s settFundingRate=%s interestRate=%s "
        "float=.12f=%s at_floor=%s",
        swap_symbol,
        out["funding_rate_raw"],
        out["sett_funding_rate_raw"],
        out["interest_rate_raw"],
        f"{rate:.12f}" if rate is not None else None,
        out["at_interest_floor"],
    )
    return out


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
    funding = extract_funding_fields(exchange, swap_symbol)

    empty_depth: dict[str, float | None] = {
        "mid": None,
        "best_bid": None,
        "best_ask": None,
        "spread_bps": None,
        "bid_notional": None,
        "ask_notional": None,
        "min_notional": None,
        "contract_size": 1.0,
    }
    perp_depth = dict(empty_depth)
    spot_depth = dict(empty_depth)

    if depth.fetch_perp_book:
        book = fetch_order_book_safe(exchange, swap_symbol, limit=depth.book_limit)
        cs = contract_size_for(exchange, swap_symbol)
        perp_depth = summarize_order_book(
            book, band_frac=depth.band_frac, contract_size=cs
        )
        logger.debug(
            "perp depth %s contract_size=%s min_notional=%s",
            swap_symbol,
            cs,
            perp_depth.get("min_notional"),
        )
    if depth.fetch_spot_book:
        book = fetch_order_book_safe(exchange, spot_symbol, limit=depth.book_limit)
        spot_depth = summarize_order_book(
            book, band_frac=depth.band_frac, contract_size=1.0
        )

    # Conservative depth for delta-neutral: thinner of the two books.
    depth_usdt = None
    p_min = perp_depth.get("min_notional")
    s_min = spot_depth.get("min_notional")
    if p_min is not None and s_min is not None:
        depth_usdt = float(min(p_min, s_min))
    elif p_min is not None:
        depth_usdt = float(p_min)
    elif s_min is not None:
        depth_usdt = float(s_min)

    # Prefer dedicated funding extract over basis (same source, richer fields).
    funding_rate = funding.get("funding_rate")
    if funding_rate is None:
        funding_rate = basis.get("funding_rate")

    return {
        "observed_at": observed_at,
        "exchange_id": exchange_id,
        "symbol": swap_symbol,
        "spot_symbol": spot_symbol,
        "funding_rate": funding_rate,
        "funding_rate_raw": funding.get("funding_rate_raw"),
        "sett_funding_rate": funding.get("sett_funding_rate"),
        "sett_funding_rate_raw": funding.get("sett_funding_rate_raw"),
        "interest_rate": funding.get("interest_rate"),
        "interest_rate_raw": funding.get("interest_rate_raw"),
        "at_interest_floor": bool(funding.get("at_interest_floor")),
        "mark_price": basis.get("mark_price"),
        "index_price": basis.get("index_price"),
        "perp_last": basis.get("mark_price"),
        "spot_last": basis.get("spot_last"),
        "quote_volume_24h": basis.get("quote_volume_24h"),
        "basis_bps_spot": basis.get("basis_bps_spot"),
        "perp_contract_size": perp_depth.get("contract_size"),
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
