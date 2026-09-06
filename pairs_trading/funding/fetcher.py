"""Fetch perpetual funding-rate history via CCXT."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from pairs_trading.data.fetcher import lookback_start_ms
from pairs_trading.funding.storage import FUNDING_COLUMNS

logger = logging.getLogger(__name__)


def funding_rows_to_frame(rows: list[dict], *, symbol: str) -> pd.DataFrame:
    """Normalize CCXT fetchFundingRateHistory rows → DataFrame."""
    if not rows:
        return pd.DataFrame(columns=FUNDING_COLUMNS)
    records = []
    for row in rows:
        ts = row.get("timestamp")
        rate = row.get("fundingRate")
        if ts is None or rate is None:
            continue
        records.append(
            {
                "timestamp": int(ts),
                "funding_rate": float(rate),
                "symbol": symbol,
            }
        )
    if not records:
        return pd.DataFrame(columns=FUNDING_COLUMNS)
    return pd.DataFrame.from_records(records)


def fetch_funding_rate_history(
    exchange,
    symbol: str,
    *,
    since_ms: int,
    until_ms: int | None = None,
    limit_per_request: int = 100,
) -> pd.DataFrame:
    """Paginate CCXT fundingRateHistory from since_ms to until_ms (inclusive-ish)."""
    until_ms = until_ms or int(datetime.now(timezone.utc).timestamp() * 1000)
    if not hasattr(exchange, "fetch_funding_rate_history"):
        raise RuntimeError(
            f"Exchange {getattr(exchange, 'id', '?')} has no fetch_funding_rate_history"
        )

    cursor = since_ms
    chunks: list[pd.DataFrame] = []
    safety = 0
    while cursor < until_ms and safety < 500:
        safety += 1
        batch = exchange.fetch_funding_rate_history(
            symbol, since=cursor, limit=limit_per_request
        )
        if not batch:
            break
        frame = funding_rows_to_frame(batch, symbol=symbol)
        if frame.empty:
            break
        # Keep only rows in requested window
        frame = frame[
            (frame["timestamp"] >= since_ms) & (frame["timestamp"] <= until_ms)
        ]
        if not frame.empty:
            chunks.append(frame)
        last_ts = int(max(int(r.get("timestamp") or 0) for r in batch))
        if last_ts <= cursor:
            break
        cursor = last_ts + 1
        if len(batch) < limit_per_request:
            break

    if not chunks:
        return pd.DataFrame(columns=FUNDING_COLUMNS)
    out = pd.concat(chunks, ignore_index=True)
    out = out.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    logger.info(
        "Fetched %d funding rows for %s (%s → %s)",
        len(out),
        symbol,
        datetime.fromtimestamp(since_ms / 1000, tz=timezone.utc).date(),
        datetime.fromtimestamp(until_ms / 1000, tz=timezone.utc).date(),
    )
    return out.reset_index(drop=True)


def default_since_ms(lookback_days: int, latest_ms: int | None) -> int:
    """Incremental sync: resume from last bar, else full lookback."""
    floor = lookback_start_ms(lookback_days)
    if latest_ms is None:
        return floor
    return max(floor, int(latest_ms) + 1)
