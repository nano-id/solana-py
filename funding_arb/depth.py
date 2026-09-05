"""Order-book depth helpers for slippage estimation (step 1).

OKX (and some other CCXT swap venues) report book amounts in *contracts*.
USDT notional = price × size × contract_size, where contract_size is
``market['contractSize']`` (OKX ``ctVal``). Spot books use contract_size=1.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _f(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def contract_size_for(exchange: Any, symbol: str) -> float:
    """Return CCXT contractSize for swaps; 1.0 for spot / unknown."""
    try:
        markets = getattr(exchange, "markets", None) or {}
        if symbol not in markets and hasattr(exchange, "market"):
            m = exchange.market(symbol)
        else:
            m = markets.get(symbol) or {}
        if not m and hasattr(exchange, "market"):
            m = exchange.market(symbol)
        cs = m.get("contractSize") if isinstance(m, dict) else None
        if cs is None and isinstance(m, dict):
            info = m.get("info") if isinstance(m.get("info"), dict) else {}
            cs = info.get("ctVal")
        val = _f(cs)
        if val is not None and val > 0:
            return float(val)
    except Exception as exc:  # noqa: BLE001
        logger.debug("contract_size lookup failed for %s: %s", symbol, exc)
    # Spot (no ':') or missing metadata → base units already.
    if ":" not in symbol:
        return 1.0
    return 1.0


def mid_from_book(bids: list, asks: list) -> float | None:
    if not bids or not asks:
        return None
    bid = _f(bids[0][0] if isinstance(bids[0], (list, tuple)) else None)
    ask = _f(asks[0][0] if isinstance(asks[0], (list, tuple)) else None)
    if bid is None or ask is None or bid <= 0 or ask <= 0:
        return None
    return 0.5 * (bid + ask)


def notional_within_band(
    levels: list,
    *,
    mid: float,
    band_frac: float,
    side: str,
    contract_size: float = 1.0,
) -> float:
    """Sum price×size×contract_size for levels within ``band_frac`` of mid.

    ``side`` is ``bid`` (exit) or ``ask`` (entry long).
    """
    if mid <= 0 or not levels:
        return 0.0
    mult = float(contract_size) if contract_size and contract_size > 0 else 1.0
    total = 0.0
    for level in levels:
        if not isinstance(level, (list, tuple)) or len(level) < 2:
            continue
        px = _f(level[0])
        sz = _f(level[1])
        if px is None or sz is None or px <= 0 or sz <= 0:
            continue
        if side == "bid":
            if px < mid * (1.0 - band_frac):
                break
        else:
            if px > mid * (1.0 + band_frac):
                break
        total += px * sz * mult
    return float(total)


def summarize_order_book(
    book: dict[str, Any] | None,
    *,
    band_frac: float,
    contract_size: float = 1.0,
) -> dict[str, float | None]:
    """Extract mid + within-band bid/ask notionals from a CCXT order book."""
    empty: dict[str, float | None] = {
        "mid": None,
        "best_bid": None,
        "best_ask": None,
        "spread_bps": None,
        "bid_notional": None,
        "ask_notional": None,
        "min_notional": None,
        "contract_size": float(contract_size) if contract_size else 1.0,
    }
    if not book:
        return empty
    bids = book.get("bids") or []
    asks = book.get("asks") or []
    mid = mid_from_book(bids, asks)
    best_bid = _f(bids[0][0]) if bids else None
    best_ask = _f(asks[0][0]) if asks else None
    spread_bps = None
    if mid and best_bid is not None and best_ask is not None and mid > 0:
        spread_bps = (best_ask - best_bid) / mid * 10_000.0
    bid_n = notional_within_band(
        bids, mid=mid or 0.0, band_frac=band_frac, side="bid", contract_size=contract_size
    )
    ask_n = notional_within_band(
        asks, mid=mid or 0.0, band_frac=band_frac, side="ask", contract_size=contract_size
    )
    min_n = min(bid_n, ask_n) if mid else 0.0
    return {
        "mid": mid,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread_bps": spread_bps,
        "bid_notional": bid_n if mid else None,
        "ask_notional": ask_n if mid else None,
        "min_notional": min_n if mid else None,
        "contract_size": float(contract_size) if contract_size else 1.0,
    }


def fetch_order_book_safe(
    exchange: Any,
    symbol: str,
    *,
    limit: int = 20,
) -> dict[str, Any] | None:
    try:
        return exchange.fetch_order_book(symbol, limit=limit)
    except Exception as exc:  # noqa: BLE001
        logger.warning("order_book failed for %s: %s", symbol, exc)
        return None
