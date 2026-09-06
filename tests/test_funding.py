from __future__ import annotations

import pandas as pd
import pytest

from pairs_trading.data.config import ExchangeConfig, StorageConfig
from pairs_trading.funding.basis import _quote_volume_24h
from pairs_trading.funding.capacity import CapacityConfig, apply_capacity_filter, passes_capacity
from pairs_trading.funding.config import FundingAppConfig, FundingConfig, load_funding_config
from pairs_trading.funding.fees import FeeConfig, fee_drag_annualized, round_trip_cost
from pairs_trading.funding.fetcher import funding_rows_to_frame
from pairs_trading.funding.harvest import build_harvest_table
from pairs_trading.funding.paper import simulate_funding_harvest
from pairs_trading.funding.persistence import funding_persistence
from pairs_trading.funding.risk import RiskConfig, harvest_side, size_position
from pairs_trading.funding.screen import annualize_funding_rate, screen_funding_store, summarize_funding_series
from pairs_trading.funding.storage import FundingStore
from pairs_trading.funding.symbols import spot_to_swap, swap_to_spot


def test_annualize_assumes_three_payments_per_day() -> None:
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
    assert loaded.iloc[0]["funding_rate"] == 0.0002
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
    assert summarize_funding_series(store.load("AAA/USDT:USDT"))["n_obs"] == 10


def test_load_funding_config_forces_swap(tmp_path) -> None:
    path = tmp_path / "funding.yaml"
    path.write_text(
        "exchange:\n  id: okx\n  market_type: spot\nfunding:\n  lookback_days: 30\n",
        encoding="utf-8",
    )
    cfg = load_funding_config(path)
    assert cfg.exchange.market_type == "swap"
    assert cfg.funding.lookback_days == 30


def test_symbol_helpers() -> None:
    assert swap_to_spot("BTC/USDT:USDT") == "BTC/USDT"
    assert spot_to_swap("BTC/USDT") == "BTC/USDT:USDT"


def test_fees_and_persistence() -> None:
    fees = FeeConfig()
    assert round_trip_cost(fees) == pytest.approx(0.003)
    assert fee_drag_annualized(fees, hold_days=30) == pytest.approx(0.0365)
    rates = pd.Series([0.001, 0.001, 0.001, -0.001, 0.001, 0.001, 0.001, 0.001])
    pers = funding_persistence(rates)
    assert 0.0 <= pers["harvest_efficiency"] <= 1.0
    assert 0.0 < pers["sign_persistence"] <= 1.0


def test_capacity_and_sizing() -> None:
    cap = CapacityConfig(min_quote_volume_24h=1_000_000)
    assert passes_capacity(2_000_000, cap)
    assert not passes_capacity(100, cap)
    df = pd.DataFrame({"quote_volume_24h": [2_000_000.0, 100.0]})
    out = apply_capacity_filter(df, cap)
    assert bool(out.iloc[0]["passes_capacity"]) is True
    assert bool(out.iloc[1]["passes_capacity"]) is False
    sized = size_position(
        "NEAR/USDT:USDT",
        funding_rate_signal=0.0001,
        quote_volume_24h=50_000_000.0,
        risk=RiskConfig(),
        capacity=cap,
    )
    assert sized.side == "short_perp_long_spot"
    assert sized.notional_usdt > 0
    assert harvest_side(-0.01) == "long_perp_short_spot"


def test_okx_quote_volume_fallback() -> None:
    tick = {
        "quoteVolume": None,
        "baseVolume": 100.0,
        "last": 2.0,
        "info": {"volCcy24h": "43243679"},
    }
    assert _quote_volume_24h(tick) == pytest.approx(43_243_679.0)
    tick2 = {"quoteVolume": None, "baseVolume": 100.0, "last": 2.0, "info": {}}
    assert _quote_volume_24h(tick2) == pytest.approx(200.0)


def test_paper_and_harvest_positive_on_synthetic(tmp_path) -> None:
    store = FundingStore(tmp_path / "data", "okx")
    n = 90
    store.save(
        "NEAR/USDT:USDT",
        pd.DataFrame(
            {
                "timestamp": [1_700_000_000_000 + i * 8 * 3600 * 1000 for i in range(n)],
                "symbol": ["NEAR/USDT:USDT"] * n,
                "funding_rate": [0.0002] * n,
            }
        ),
    )
    config = FundingAppConfig(
        exchange=ExchangeConfig(id="okx", market_type="swap"),
        storage=StorageConfig(root=tmp_path / "data"),
        funding=FundingConfig(default_symbols=["NEAR/USDT:USDT"]),
    )
    table = build_harvest_table(config, exchange_id="okx")
    assert not table.empty
    assert float(table.iloc[0]["net_ann"]) > 0
    assert bool(table.iloc[0]["positive_net"]) is True

    hist = store.load("NEAR/USDT:USDT")
    result = simulate_funding_harvest(
        hist["funding_rate"],
        symbol="NEAR/USDT:USDT",
        notional_usdt=2500.0,
        fee_config=FeeConfig(),
        settlements_per_day=3.0,
    )
    assert result.net_pnl > 0
    assert result.ann_return > 0
