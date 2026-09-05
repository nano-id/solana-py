"""Unit tests for funding_arb step 1 (depth math + storage; no live exchange)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from funding_arb.depth import mid_from_book, notional_within_band, summarize_order_book
from funding_arb.loop import run_collect_loop
from funding_arb.storage import save_snapshot_batch, snapshot_summary


def test_notional_within_band_bids() -> None:
    mid = 100.0
    # Within 20 bps: bids >= 99.8
    bids = [[99.95, 10], [99.85, 5], [99.0, 100]]  # last outside band
    n = notional_within_band(bids, mid=mid, band_frac=0.002, side="bid")
    assert abs(n - (99.95 * 10 + 99.85 * 5)) < 1e-6


def test_summarize_order_book_spread_and_min() -> None:
    book = {
        "bids": [[100.0, 2.0], [99.9, 3.0]],
        "asks": [[100.2, 1.5], [100.3, 4.0]],
    }
    s = summarize_order_book(book, band_frac=0.01)
    assert s["mid"] == 100.1
    assert s["spread_bps"] is not None and s["spread_bps"] > 0
    assert s["min_notional"] == min(s["bid_notional"], s["ask_notional"])


def test_mid_empty_book() -> None:
    assert mid_from_book([], [[1, 1]]) is None


def test_save_snapshot_batch_latest(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "observed_at": "2026-01-01T00:00:00+00:00",
                "exchange_id": "okx",
                "symbol": "BTC/USDT:USDT",
                "funding_rate": 0.0001,
                "spot_last": 42000.0,
                "perp_last": 42010.0,
                "depth_usdt": 50000.0,
            }
        ]
    )
    paths = save_snapshot_batch(frame, root=tmp_path, exchange_id="okx", mode="both")
    assert paths
    latest = tmp_path / "okx" / "funding_arb" / "snapshots" / "latest.csv"
    assert latest.is_file()
    loaded = pd.read_csv(latest)
    assert loaded.iloc[0]["symbol"] == "BTC/USDT:USDT"
    summary = snapshot_summary(frame)
    assert summary["rows"] == 1
    assert summary["with_funding"] == 1


def test_collect_loop_respects_max_cycles(monkeypatch) -> None:
    """Loop orchestration without hitting a real exchange."""
    from funding_arb import loop as loop_mod
    from pairs_trading.data.exchange import ConnectedExchange

    class _DummyEx:
        pass

    calls = {"n": 0}

    def fake_cycle(config, *, symbols=None, connected=None, persist=True, per_symbol_ticks=True):
        calls["n"] += 1
        frame = pd.DataFrame([{"symbol": "BTC/USDT:USDT", "funding_rate": 0.0001}])
        conn = connected or ConnectedExchange(_DummyEx(), "okx", ("okx",))
        return frame, conn, []

    monkeypatch.setattr(loop_mod, "run_collect_cycle", fake_cycle)
    monkeypatch.setattr(
        loop_mod,
        "connect_collector",
        lambda config: ConnectedExchange(_DummyEx(), "okx", ("okx",)),
    )
    sleeps: list[float] = []
    n = run_collect_loop(
        max_cycles=3,
        persist=False,
        sleep_fn=lambda s: sleeps.append(s),
    )
    assert n == 3
    assert calls["n"] == 3
    assert len(sleeps) == 2  # no sleep after final cycle
