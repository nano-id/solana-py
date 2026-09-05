"""Historical OHLCV pagination via CCXT.

Look-ahead controls:
- Only closed candles are kept when `closed_only=True` (default).
- Fetch windows are bounded by explicit since/until timestamps.
- Incremental updates start from last stored closed bar, not "now" guesses.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

import ccxt
import pandas as pd

from pairs_trading.data.config import OhlcvConfig
from pairs_trading.data.storage import ohlcv_rows_to_frame

logger = logging.getLogger(__name__)


def timeframe_to_milliseconds(exchange: ccxt.Exchange, timeframe: str) -> int:
    ms = exchange.parse_timeframe(timeframe) * 1000
    if ms <= 0:
        raise ValueError(f"Invalid timeframe: {timeframe}")
    return int(ms)


def ms_to_utc(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def lookback_start_ms(lookback_days: int, *, now: datetime | None = None) -> int:
    now = now or datetime.now(timezone.utc)
    start = now - timedelta(days=lookback_days)
    return int(start.timestamp() * 1000)


def drop_unclosed_candle(
    df: pd.DataFrame,
    *,
    timeframe_ms: int,
    now_ms: int | None = None,
) -> pd.DataFrame:
    """Remove the in-progress candle so features never see partial bars."""
    if df.empty:
        return df
    now_ms = now_ms if now_ms is not None else int(datetime.now(timezone.utc).timestamp() * 1000)
    # A candle that opened at ts closes at ts + timeframe_ms.
    closed_mask = df["timestamp"] + timeframe_ms <= now_ms
    return df.loc[closed_mask].reset_index(drop=True)


def fetch_ohlcv_range(
    exchange: ccxt.Exchange,
    symbol: str,
    ohlcv_config: OhlcvConfig,
    *,
    since_ms: int,
    until_ms: int | None = None,
    closed_only: bool = True,
    sleep_on_empty: float = 0.0,
) -> pd.DataFrame:
    """Paginate OHLCV from since_ms up to until_ms (exclusive of open candle)."""
    timeframe = ohlcv_config.timeframe
    timeframe_ms = timeframe_to_milliseconds(exchange, timeframe)
    limit = ohlcv_config.limit_per_request
    until_ms = until_ms or int(datetime.now(timezone.utc).timestamp() * 1000)

    all_rows: list[list[float | int]] = []
    cursor = since_ms
    safety_iters = 0
    max_iters = 10_000

    while cursor < until_ms and safety_iters < max_iters:
        safety_iters += 1
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
        except ccxt.NetworkError as exc:
            logger.warning("Network error fetching %s: %s — retrying once", symbol, exc)
            time.sleep(exchange.rateLimit / 1000 if exchange.rateLimit else 1.0)
            batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)

        if not batch:
            if sleep_on_empty > 0:
                time.sleep(sleep_on_empty)
            break

        all_rows.extend(batch)
        last_ts = int(batch[-1][0])
        next_cursor = last_ts + timeframe_ms
        if next_cursor <= cursor:
            # Exchange returned a stuck cursor; stop to avoid infinite loop.
            break
        cursor = next_cursor

        # Partial final page → done.
        if len(batch) < limit:
            break

    df = ohlcv_rows_to_frame(all_rows)
    if df.empty:
        return df

    # Keep bars whose open is in [since_ms, until_ms).
    df = df[(df["timestamp"] >= since_ms) & (df["timestamp"] < until_ms)].copy()
    if closed_only:
        df = drop_unclosed_candle(df, timeframe_ms=timeframe_ms, now_ms=until_ms)
    return df.reset_index(drop=True)
