"""Unit tests for config + storage (no network)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pairs_trading.data.config import DataConfig, load_config
from pairs_trading.data.storage import ParquetStore, ohlcv_rows_to_frame


def test_load_default_config():
    cfg = load_config(Path("config/default.yaml"))
    assert cfg.exchange.id == "binance"
    assert cfg.universe.top_n == 30
    assert cfg.ohlcv.timeframe == "1h"
    assert "USDT" in cfg.universe.exclude_bases


def test_data_config_defaults():
    cfg = DataConfig()
    assert cfg.exchange.quote == "USDT"
    assert cfg.storage.root == Path("data")


def test_parquet_store_upsert_dedupes(tmp_path: Path):
    store = ParquetStore(tmp_path, "binance", "1h")
    rows1 = [
        [1_700_000_000_000, 1, 2, 0.5, 1.5, 100],
        [1_700_003_600_000, 1.5, 2.5, 1.0, 2.0, 110],
    ]
    rows2 = [
        # Revised close for overlapping timestamp
        [1_700_003_600_000, 1.5, 2.5, 1.0, 2.2, 115],
        [1_700_007_200_000, 2.0, 3.0, 1.8, 2.8, 120],
    ]
    store.upsert("BTC/USDT", ohlcv_rows_to_frame(rows1))
    merged = store.upsert("BTC/USDT", ohlcv_rows_to_frame(rows2))
    assert len(merged) == 3
    mid = merged.loc[merged["timestamp"] == 1_700_003_600_000].iloc[0]
    assert mid["close"] == pytest.approx(2.2)
    assert mid["volume"] == pytest.approx(115)
    assert store.latest_timestamp_ms("BTC/USDT") == 1_700_007_200_000


def test_ohlcv_rows_empty():
    df = ohlcv_rows_to_frame([])
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert df.empty
