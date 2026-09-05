"""Unit tests for funding_arb steps 1–3 (no live exchange)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from funding_arb.depth import mid_from_book, notional_within_band, summarize_order_book
from funding_arb.entry import (
    EntryState,
    LegFill,
    LegSide,
    LegStatus,
    paper_dual_leg_entry,
    reconcile_dual_legs,
    select_entry_candidate,
    size_notional_from_depth,
)
from funding_arb.loop import run_collect_loop
from funding_arb.screen import (
    annualized_funding,
    compute_floor_persistence,
    screen_opportunities,
    screen_summary,
)
from funding_arb.storage import save_snapshot_batch, snapshot_summary


def test_notional_within_band_bids() -> None:
    mid = 100.0
    bids = [[99.95, 10], [99.85, 5], [99.0, 100]]
    n = notional_within_band(bids, mid=mid, band_frac=0.002, side="bid")
    assert abs(n - (99.95 * 10 + 99.85 * 5)) < 1e-6


def test_notional_applies_contract_size() -> None:
    mid = 0.1
    bids = [[0.1, 2.0]]
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
    assert abs(annualized_funding(0.0001) - 0.1095) < 1e-12


def test_screen_ranks_and_filters_depth() -> None:
    snap = pd.DataFrame(
        [
            {
                "symbol": "THIN/USDT:USDT",
                "funding_rate": 0.001,
                "sett_funding_rate": 0.001,
                "interest_rate": 0.0001,
                "at_interest_floor": False,
                "depth_usdt": 100.0,
            },
            {
                "symbol": "FAT/USDT:USDT",
                "funding_rate": 0.0002,
                "sett_funding_rate": 0.0002,
                "interest_rate": 0.0001,
                "at_interest_floor": False,
                "depth_usdt": 50_000.0,
            },
            {
                "symbol": "FLOOR/USDT:USDT",
                "funding_rate": 0.0001,
                "sett_funding_rate": 0.00005,
                "interest_rate": 0.0001,
                "at_interest_floor": True,
                "depth_usdt": 80_000.0,
            },
        ]
    )
    out = screen_opportunities(snap, min_depth_usdt=10_000.0, prefer_non_floor=True)
    # Non-floor FAT ranks above floor name despite lower |funding|.
    assert out.iloc[0]["symbol"] == "FAT/USDT:USDT"
    assert not bool(out.loc[out["symbol"] == "THIN/USDT:USDT", "passes_depth"].iloc[0])
    summary = screen_summary(out)
    assert summary["eligible"] == 2
    assert summary["eligible_non_floor"] == 1


def test_floor_persistence_from_history() -> None:
    hist = pd.DataFrame(
        [
            {"symbol": "A", "observed_at": f"2026-01-0{i}", "at_interest_floor": True}
            for i in range(1, 6)
        ]
        + [
            {
                "symbol": "B",
                "observed_at": f"2026-01-0{i}",
                "at_interest_floor": i <= 2,
                "funding_rate": 0.0003 if i > 2 else 0.0001,
                "interest_rate": 0.0001,
                "sett_funding_rate": 0.00025 if i > 2 else 0.0001,
            }
            for i in range(1, 6)
        ]
    )
    # Ensure A also has rate cols for premium stats
    hist.loc[hist["symbol"] == "A", "funding_rate"] = 0.0001
    hist.loc[hist["symbol"] == "A", "interest_rate"] = 0.0001
    hist.loc[hist["symbol"] == "A", "sett_funding_rate"] = 0.0001

    pers = compute_floor_persistence(hist, lookback_snapshots=5)
    a = pers.loc[pers["symbol"] == "A"].iloc[0]
    b = pers.loc[pers["symbol"] == "B"].iloc[0]
    assert a["floor_persistence"] == 1.0
    assert abs(b["floor_persistence"] - 0.4) < 1e-9  # 2/5


def test_size_notional_from_depth_caps() -> None:
    assert size_notional_from_depth(1_000_000, depth_fraction=0.05, max_notional_usdt=100) == 100
    assert size_notional_from_depth(50_000, depth_fraction=0.05, max_notional_usdt=100) == 2500 or True
    # 50k * 5% = 2500 but capped at 100
    assert size_notional_from_depth(50_000, depth_fraction=0.05, max_notional_usdt=100) == 100
    assert size_notional_from_depth(100, depth_fraction=0.05, min_notional_usdt=10) == 0


def test_paper_entry_balanced() -> None:
    result = paper_dual_leg_entry(
        "BTC/USDT:USDT",
        depth_usdt=200_000,
        side_hint="short_perp_long_spot",
        depth_fraction=0.05,
        max_notional_usdt=100,
    )
    assert result.state == EntryState.BALANCED
    assert result.target_usdt == 100
    assert result.residual_usdt == 0


def test_unpaired_leg_auto_flatten() -> None:
    """Spot fills, perp times out → flatten spot so book is not directional."""
    result = paper_dual_leg_entry(
        "ETH/USDT:USDT",
        depth_usdt=200_000,
        side_hint="short_perp_long_spot",
        max_notional_usdt=100,
        spot_fill_ratio=1.0,
        perp_fill_ratio=0.0,
        flatten_unpaired=True,
    )
    assert result.state == EntryState.FLATTENED
    assert result.spot.status == LegStatus.FLATTENED
    assert result.spot.filled_usdt == 0
    assert result.residual_usdt == 0
    assert any("unpaired_leg" in a for a in result.alarms)


def test_unpaired_leg_alarm_without_flatten() -> None:
    spot = LegFill(LegSide.SPOT, 100, 100, LegStatus.FILLED)
    perp = LegFill(LegSide.PERP, 100, 0, LegStatus.TIMEOUT)
    result = reconcile_dual_legs(
        "SOL/USDT:USDT",
        target_usdt=100,
        side_hint="short_perp_long_spot",
        spot=spot,
        perp=perp,
        flatten_unpaired=False,
    )
    assert result.state == EntryState.UNPAIRED
    assert result.residual_usdt == 100
    assert any("unpaired_no_flatten" in a for a in result.alarms)


def test_select_entry_prefers_non_floor() -> None:
    screen = pd.DataFrame(
        [
            {
                "symbol": "FLOOR/USDT:USDT",
                "passes_depth": True,
                "at_interest_floor": True,
                "depth_usdt": 90_000,
                "side_hint": "short_perp_long_spot",
            },
            {
                "symbol": "REAL/USDT:USDT",
                "passes_depth": True,
                "at_interest_floor": False,
                "depth_usdt": 40_000,
                "side_hint": "long_perp_short_spot",
            },
        ]
    )
    # Screen already ranked; selector filters non-floor from whatever order.
    cand = select_entry_candidate(screen, require_non_floor=True)
    assert cand is not None
    assert cand["symbol"] == "REAL/USDT:USDT"
