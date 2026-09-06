"""Tests for OHLC→synth BBO event path."""

from __future__ import annotations

import pandas as pd

from market_making.config import MarketMakingConfig
from market_making.hist_bbo import synth_bbo_from_ohlc, run_month_events_paper


def test_synth_bbo_from_ohlc_counts() -> None:
    rows = []
    t0 = 1_700_000_000_000
    for i in range(3):
        rows.append(
            {
                "timestamp": t0 + i * 3_600_000,
                "open": 100.0 + i,
                "high": 101.0 + i,
                "low": 99.0 + i,
                "close": 100.5 + i,
                "volume": 1000.0,
            }
        )
    df = pd.DataFrame(rows)
    ev = synth_bbo_from_ohlc(df, "BTC/USDT", steps_per_bar=10, seed=1)
    assert len(ev) == 30
    assert all(e.valid for e in ev)
    assert ev[0].ts_ms < ev[-1].ts_ms


def test_month_events_paper_smoke() -> None:
    cfg = MarketMakingConfig(
        symbols=["BTC/USDT", "ETH/USDT"],
        commission_pct=0.07,
        storage_root="data",
        exchange_id="okx",
    )
    # Tiny synthetic path via run on real Mar 2025 but we smoke via synth only:
    # use build through run_month_events with steps=2 if data present.
    result, rows, meta = run_month_events_paper(
        cfg,
        year=2025,
        month=3,
        timeframe="1h",
        steps_per_bar=2,
        seed=3,
        fill_prob=0.25,
        adverse_frac=0.55,
    )
    assert meta["n_events"] > 0
    assert result.interpretation_allowed is False
    assert result.n_events == meta["n_events"] or result.n_events <= meta["n_events"]
