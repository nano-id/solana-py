"""Event-driven MM tests (BBO stream, not candles)."""

from __future__ import annotations

import time

from market_making.config import MarketMakingConfig
from market_making.feed import BookSnapshot, snapshot_from_order_book
from market_making.realtime import run_realtime_paper


def _synth_events(n_ticks: int = 40, symbols: list[str] | None = None):
    symbols = symbols or ["AAA/USDT", "BBB/USDT", "CCC/USDT"]
    mids = {s: 50.0 + i * 10 for i, s in enumerate(symbols)}
    t0 = int(time.time() * 1000)
    out = []
    for i in range(n_ticks):
        for s in symbols:
            mids[s] *= 1.0 + 0.0002 * ((i + len(s)) % 5 - 2)
            m = mids[s]
            sp = m * 0.0003
            out.append(
                BookSnapshot(
                    symbol=s,
                    ts_ms=t0 + i * 1000,
                    bid=m - sp / 2,
                    ask=m + sp / 2,
                    bid_size=2.0,
                    ask_size=2.0,
                    mid=m,
                    spread_bps=3.0,
                    exchange_id="mock",
                )
            )
    return out


def test_snapshot_from_order_book() -> None:
    book = {"bids": [[100.0, 1.5]], "asks": [[100.2, 2.0]], "timestamp": 1_700_000_000_000}
    snap = snapshot_from_order_book("BTC/USDT", book, exchange_id="okx")
    assert snap is not None
    assert snap.valid
    assert snap.mid == 100.1
    assert snap.spread_bps > 0


def test_realtime_hold_time_is_seconds_not_bars() -> None:
    cfg = MarketMakingConfig(
        commission_pct=0.07,
        symbols=["AAA/USDT", "BBB/USDT", "CCC/USDT"],
    )
    result, rows = run_realtime_paper(
        cfg, _synth_events(50), seed=2, initial_equity=1000.0, poll_interval_sec=1.0
    )
    assert result.n_events > 0
    assert result.interpretation_allowed is False
    if result.n_holds > 0:
        assert result.hold_seconds_mean is not None
        # Synthetic ticks are 1s apart; holds should be on the order of seconds, not hours.
        assert result.hold_seconds_mean < 3600.0
        assert result.hold_seconds_max is not None
        assert result.hold_seconds_max < 3600.0


def test_realtime_single_wallet_lock() -> None:
    cfg = MarketMakingConfig(
        commission_pct=0.07,
        symbols=["AAA/USDT", "BBB/USDT", "CCC/USDT"],
    )
    _result, rows = run_realtime_paper(
        cfg, _synth_events(60), seed=5, initial_equity=1000.0
    )
    cur = None
    for row in rows:
        if row["action"] == "enter":
            assert cur is None
            cur = row["active_symbol"]
        elif row["action"] == "close":
            assert cur is not None
            cur = None
        elif row["action"] == "manage":
            assert row["active_symbol"] == cur
