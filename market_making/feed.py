"""Live spot BBO / order-book feed (event source for MM).

Market making must react to the book, not to closed candles.
This module polls CCXT spot order books and emits mid/BBO snapshots.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Iterator

from pairs_trading.data.config import ExchangeConfig
from pairs_trading.data.exchange import ConnectedExchange, connect_with_fallback

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BookSnapshot:
    """Top-of-book event (anlık veri birimi)."""

    symbol: str
    ts_ms: int
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    mid: float
    spread_bps: float
    exchange_id: str

    @property
    def valid(self) -> bool:
        return (
            self.bid > 0
            and self.ask > self.bid
            and self.mid > 0
            and np_isfinite(self.mid)
        )


def np_isfinite(x: float) -> bool:
    return x == x and x not in (float("inf"), float("-inf"))


def connect_spot_exchange(exchange_id: str = "okx") -> ConnectedExchange:
    """Connect to a spot venue (OKX primary, with geo fallbacks)."""
    cfg = ExchangeConfig(
        id=exchange_id,
        fallbacks=["bybit", "binance", "okx"],
        market_type="spot",
        enable_rate_limit=True,
    )
    return connect_with_fallback(cfg)


def snapshot_from_order_book(
    symbol: str,
    book: dict[str, Any],
    *,
    exchange_id: str,
    ts_ms: int | None = None,
) -> BookSnapshot | None:
    """Build a BBO snapshot from a CCXT order book payload."""
    bids = book.get("bids") or []
    asks = book.get("asks") or []
    if not bids or not asks:
        return None
    try:
        bid_px, bid_sz = float(bids[0][0]), float(bids[0][1])
        ask_px, ask_sz = float(asks[0][0]), float(asks[0][1])
    except (TypeError, ValueError, IndexError):
        return None
    if bid_px <= 0 or ask_px <= bid_px:
        return None
    mid = 0.5 * (bid_px + ask_px)
    spread_bps = 1e4 * (ask_px - bid_px) / mid
    raw_ts = book.get("timestamp")
    if ts_ms is None:
        ts_ms = int(raw_ts) if raw_ts is not None else int(time.time() * 1000)
    return BookSnapshot(
        symbol=symbol,
        ts_ms=int(ts_ms),
        bid=bid_px,
        ask=ask_px,
        bid_size=bid_sz,
        ask_size=ask_sz,
        mid=mid,
        spread_bps=float(spread_bps),
        exchange_id=exchange_id,
    )


def fetch_book_snapshot(
    exchange: Any,
    symbol: str,
    *,
    exchange_id: str,
    limit: int = 5,
) -> BookSnapshot | None:
    book = exchange.fetch_order_book(symbol, limit=limit)
    return snapshot_from_order_book(symbol, book, exchange_id=exchange_id)


def poll_book_events(
    symbols: list[str],
    *,
    exchange_id: str = "okx",
    poll_interval_sec: float = 1.0,
    book_limit: int = 5,
    max_cycles: int | None = None,
    exchange: Any | None = None,
) -> Iterator[BookSnapshot]:
    """Yield BBO snapshots by polling spot order books.

    This is still polling (REST), not a websocket stream — but the *decision
    clock* is the book update, not a 4h candle close. Websocket can replace
    the transport later without changing the MM event loop.
    """
    connected = None
    if exchange is None:
        connected = connect_spot_exchange(exchange_id)
        exchange = connected.exchange
        exchange_id = connected.exchange_id
        logger.info("Realtime feed on %s for %s", exchange_id, symbols)

    cycles = 0
    while max_cycles is None or cycles < max_cycles:
        cycles += 1
        for sym in symbols:
            try:
                snap = fetch_book_snapshot(
                    exchange, sym, exchange_id=exchange_id, limit=book_limit
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("book fetch failed %s: %s", sym, exc)
                continue
            if snap is not None and snap.valid:
                yield snap
        if max_cycles is None or cycles < max_cycles:
            time.sleep(max(poll_interval_sec, 0.0))
