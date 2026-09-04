from __future__ import annotations

import pandas as pd
import pytest

from pairs_trading.data.config import ExchangeConfig, StorageConfig
from pairs_trading.funding.config import FundingAppConfig, FundingConfig, load_funding_config
from pairs_trading.funding.fetcher import funding_rows_to_frame
from pairs_trading.funding.screen import annualize_funding_rate, screen_funding_store, summarize_funding_series
from pairs_trading.funding.storage import FundingStore


def test_annualize_assumes_three_payments_per_day() -> None:
    # 0.01% per 8h → 0.0001 * 3 * 365 = 10.95%
    assert annualize_funding_rate(0.0001) == pytest.approx(0.1095)


def test_funding_rows_to_frame() -> None:
    raw = [
        {"timestamp": 2_000, "fundingRate": 0.0002},
        {"timestamp": 1_000, "fundingRate": 0.0001},
        {"timestamp": None, "fundingRate": 0.9},
        {"timestamp": 3_000, "fundingRate": None},
    ]
    frame = funding_rows_to_frame(raw, symbol="BTC/USDT:USDT")
    assert list(frame["timestamp"]) == [2000, 1000]
    assert list(frame["funding_rate"]) == [0.0002, 0.0001]
    assert (frame["symbol"] == "BTC/USDT:USDT").all()


def test_funding_store_roundtrip(tmp_path) -> None:
    store = FundingStore(tmp_path / "data", "okx")
    frame = pd.DataFrame(
        {
            "timestamp": [1_700_000_000_000, 1_700_028_800_000, 1_700_000_000_000],
            "symbol": ["BTC/USDT:USDT"] * 3,
            "funding_rate": [0.0001, -0.00005, 0.0002],
        }
    )
    store.save("BTC/USDT:USDT", frame)
    loaded = store.load("BTC/USDT:USDT")
    assert len(loaded) == 2
    assert loaded.iloc[0]["funding_rate"] == 0.0002  # keep=last on duplicate ts
    assert store.latest_timestamp_ms("BTC/USDT:USDT") == 1_700_028_800_000


def test_summarize_and_screen_ranks_by_abs_ann(tmp_path) -> None:
    store = FundingStore(tmp_path / "data", "okx")
    for sym, rate in [("AAA/USDT:USDT", 0.001), ("BBB/USDT:USDT", 0.0001)]:
        n = 10
        store.save(
            sym,
            pd.DataFrame(
                {
                    "timestamp": [1_700_000_000_000 + i * 8 * 3600 * 1000 for i in range(n)],
                    "symbol": [sym] * n,
                    "funding_rate": [rate] * n,
                }
            ),
        )
    config = FundingAppConfig(
        exchange=ExchangeConfig(id="okx", market_type="swap"),
        storage=StorageConfig(root=tmp_path / "data"),
        funding=FundingConfig(default_symbols=[]),
    )
    ranked = screen_funding_store(config, exchange_id="okx")
    assert ranked.iloc[0]["symbol"] == "AAA/USDT:USDT"
    assert float(ranked.iloc[0]["ann_mean"]) == pytest.approx(1.095)
    stats = summarize_funding_series(store.load("AAA/USDT:USDT"))
    assert stats["n_obs"] == 10


def test_load_funding_config_forces_swap(tmp_path) -> None:
    path = tmp_path / "funding.yaml"
    path.write_text(
        "exchange:\n  id: okx\n  market_type: spot\nfunding:\n  lookback_days: 30\n",
        encoding="utf-8",
    )
    cfg = load_funding_config(path)
    assert cfg.exchange.market_type == "swap"
    assert cfg.funding.lookback_days == 30
