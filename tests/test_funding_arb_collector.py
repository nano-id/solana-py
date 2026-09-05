"""Unit tests for funding_arb steps 1–2 (no live exchange)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from funding_arb.depth import mid_from_book, notional_within_band, summarize_order_book
from funding_arb.loop import run_collect_loop
from funding_arb.screen import annualized_funding, screen_opportunities, screen_summary
from funding_arb.storage import save_snapshot_batch, snapshot_summary


def test_notional_within_band_bids() -> None:
    mid = 100.0
    bids = [[99.95, 10], [99.85, 5], [99.0, 100]]  # last outside 20 bps
    n = notional_within_band(bids, mid=mid, band_frac=0.002, side="bid")
    assert abs(n - (99.95 * 10 + 99.85 * 5)) < 1e-6


def test_notional_applies_contract_size() -> None:
    """OKX-style: size is contracts; USDT = px × sz × ctVal."""
    mid = 0.1
    bids = [[0.1, 2.0]]  # 2 contracts
    raw = notional_within_band(bids, mid=mid, band_frac=0.01, side="bid", contract_size=1.0)
    scaled = notional_within_band(
        bids, mid=mid, band_frac=0.01, side="bid", contract_size=1000.0
    )
    assert abs(raw - 0.2) < 1e-9
    assert abs(scaled - 200.0) < 1e-9


def test_summarize_order_book_spread_and_min() -> None:
    book = {
        "bids": [[100.0, 2.0], [99.9, 3.0]],
        "asks": [[100.2, 1.5], [100.3, 4.0]],
    }
    s = summarize_order_book(book, band_frac=0.01, contract_size=1.0)
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
    assert len(sleeps) == 2


def test_annualized_funding_8h_cycle() -> None:
    # 0.0001 per 8h → 0.0001 * 3 * 365 = 0.1095
    assert abs(annualized_funding(0.0001) - 0.1095) < 1e-12


def test_screen_ranks_and_filters_depth() -> None:
    snap = pd.DataFrame(
        [
            {
                "symbol": "THIN/USDT:USDT",
                "funding_rate": 0.001,
                "sett_funding_rate": 0.001,
                "at_interest_floor": False,
                "depth_usdt": 100.0,
            },
            {
                "symbol": "FAT/USDT:USDT",
                "funding_rate": 0.0002,
                "sett_funding_rate": 0.0002,
                "at_interest_floor": False,
                "depth_usdt": 50_000.0,
            },
            {
                "symbol": "FLOOR/USDT:USDT",
                "funding_rate": 0.0001,
                "sett_funding_rate": 0.00005,
                "at_interest_floor": True,
                "depth_usdt": 80_000.0,
            },
        ]
    )
    out = screen_opportunities(snap, min_depth_usdt=10_000.0)
    assert list(out["symbol"])[:2] == ["FAT/USDT:USDT", "FLOOR/USDT:USDT"]
    assert not bool(out.loc[out["symbol"] == "THIN/USDT:USDT", "passes_depth"].iloc[0])
    assert bool(out.loc[out["symbol"] == "FAT/USDT:USDT", "passes_depth"].iloc[0])
    # FAT has higher |ann| among eligible → rank 1
    assert int(out.loc[out["symbol"] == "FAT/USDT:USDT", "rank"].iloc[0]) == 1
    summary = screen_summary(out)
    assert summary["eligible"] == 2
    assert summary["at_interest_floor"] == 1
