"""Tests for fetcher look-ahead guards and universe filtering (mocked)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pairs_trading.data.config import ExchangeConfig, OhlcvConfig, UniverseConfig
from pairs_trading.data.fetcher import drop_unclosed_candle, fetch_ohlcv_range
from pairs_trading.data.universe import (
    load_universe_snapshot,
    save_universe_snapshot,
    select_universe,
)


class _FakeExchange:
    rateLimit = 0

    def __init__(self, batches: list[list[list[float | int]]]):
        self._batches = list(batches)
        self.calls: list[dict] = []

    def parse_timeframe(self, timeframe: str) -> int:
        mapping = {"1m": 60, "1h": 3600, "1d": 86400}
        return mapping[timeframe]

    def fetch_ohlcv(self, symbol, timeframe="1h", since=None, limit=None):
        self.calls.append(
            {"symbol": symbol, "timeframe": timeframe, "since": since, "limit": limit}
        )
        if not self._batches:
            return []
        return self._batches.pop(0)

    def load_markets(self):
        return {
            "BTC/USDT": {
                "symbol": "BTC/USDT",
                "base": "BTC",
                "quote": "USDT",
                "active": True,
                "spot": True,
            },
            "ETH/USDT": {
                "symbol": "ETH/USDT",
                "base": "ETH",
                "quote": "USDT",
                "active": True,
                "spot": True,
            },
            "USDC/USDT": {
                "symbol": "USDC/USDT",
                "base": "USDC",
                "quote": "USDT",
                "active": True,
                "spot": True,
            },
            "LOW/USDT": {
                "symbol": "LOW/USDT",
                "base": "LOW",
                "quote": "USDT",
                "active": True,
                "spot": True,
            },
            "ADA/BTC": {
                "symbol": "ADA/BTC",
                "base": "ADA",
                "quote": "BTC",
                "active": True,
                "spot": True,
            },
        }

    def fetch_tickers(self):
        return {
            "BTC/USDT": {"quoteVolume": 2_000_000_000, "last": 60000},
            "ETH/USDT": {"quoteVolume": 1_000_000_000, "last": 3000},
            "USDC/USDT": {"quoteVolume": 5_000_000_000, "last": 1},
            "LOW/USDT": {"quoteVolume": 1000, "last": 1},
            "ADA/BTC": {"quoteVolume": 9_000_000_000, "last": 0.0001},
        }


def test_drop_unclosed_candle():
    tf_ms = 3_600_000
    # Candle opens at t=0; closes at t=3600000
    df = pd.DataFrame(
        {
            "timestamp": [0, 3_600_000, 7_200_000],
            "open": [1, 2, 3],
            "high": [1, 2, 3],
            "low": [1, 2, 3],
            "close": [1, 2, 3],
            "volume": [1, 1, 1],
        }
    )
    # "Now" is mid-way through the third candle → only first two are closed.
    out = drop_unclosed_candle(df, timeframe_ms=tf_ms, now_ms=7_200_000 + 1_000)
    assert list(out["timestamp"]) == [0, 3_600_000]


def test_fetch_ohlcv_range_paginates_and_filters():
    tf_ms = 3_600_000
    batch1 = [[i * tf_ms, 1, 1, 1, 1, 1] for i in range(3)]
    batch2 = [[(3 + i) * tf_ms, 1, 1, 1, 1, 1] for i in range(2)]
    exchange = _FakeExchange([batch1, batch2])
    cfg = OhlcvConfig(timeframe="1h", lookback_days=7, limit_per_request=3)
    # until mid last candle so last open candle dropped
    until = 5 * tf_ms + 100
    df = fetch_ohlcv_range(
        exchange,
        "BTC/USDT",
        cfg,
        since_ms=0,
        until_ms=until,
        closed_only=True,
    )
    assert len(exchange.calls) == 2
    assert list(df["timestamp"]) == [0, tf_ms, 2 * tf_ms, 3 * tf_ms, 4 * tf_ms]


def test_select_universe_filters_stables_and_volume(tmp_path: Path):
    exchange = _FakeExchange([])
    snap = select_universe(
        exchange,
        ExchangeConfig(id="binance", quote="USDT", market_type="spot"),
        UniverseConfig(top_n=10, min_quote_volume=5_000_000, exclude_bases=["USDC", "USDT"]),
        as_of=datetime(2024, 1, 15, tzinfo=timezone.utc),
    )
    symbols = snap.symbols
    assert symbols == ["BTC/USDT", "ETH/USDT"]
    assert snap.members[0].rank == 1
    path = tmp_path / "universe.json"
    save_universe_snapshot(snap, path)
    loaded = load_universe_snapshot(path)
    assert loaded.symbols == symbols
    assert loaded.as_of.startswith("2024-01-15")
