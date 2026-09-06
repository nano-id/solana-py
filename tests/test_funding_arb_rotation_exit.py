"""Unit tests for step 4 rotation + step 5 exits + control ordering."""

from __future__ import annotations

import pandas as pd

from funding_arb.control import CycleAction, run_control_cycle
from funding_arb.exit_rules import ExitAction, ExitReason, decide_exit
from funding_arb.position import PaperPosition
from funding_arb.rotation import RotationAction, decide_rotation


def _screen(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _pos(
    symbol: str = "SOL/USDT:USDT",
    *,
    side: str = "long_perp_short_spot",
    premium: float = -0.00005,
) -> PaperPosition:
    return PaperPosition(
        symbol=symbol,
        side_hint=side,
        notional_usdt=100.0,
        entry_funding_rate=premium,
        entry_premium_rate=premium,
        entry_at_floor=False,
    )


def test_rotation_hold_when_below_3x_among_non_floor() -> None:
    pos = _pos(premium=-0.00005)
    screen = _screen(
        [
            {
                "symbol": "SOL/USDT:USDT",
                "funding_rate": -0.00005,
                "premium_rate": -0.00005,
                "at_interest_floor": False,
                "passes_depth": True,
                "depth_usdt": 400_000,
                "side_hint": "long_perp_short_spot",
            },
            {
                "symbol": "BTC/USDT:USDT",
                "funding_rate": -0.00008,
                "premium_rate": -0.00010,
                "at_interest_floor": False,
                "passes_depth": True,
                "depth_usdt": 200_000,
                "side_hint": "long_perp_short_spot",
            },
            {
                "symbol": "ETH/USDT:USDT",
                "funding_rate": 0.0001,
                "premium_rate": 0.0,
                "at_interest_floor": True,
                "passes_depth": True,
                "depth_usdt": 300_000,
                "side_hint": "short_perp_long_spot",
            },
        ]
    )
    d = decide_rotation(pos, screen, min_edge_multiple=3.0)
    assert d.action == RotationAction.HOLD
    assert d.ratio < 3.0
    assert d.candidate is not None
    assert d.candidate["symbol"] == "BTC/USDT:USDT"
    assert not bool(d.candidate["at_interest_floor"])


def test_rotation_triggers_at_3x_non_floor_premium() -> None:
    pos = _pos(premium=-0.00005)
    screen = _screen(
        [
            {
                "symbol": "SOL/USDT:USDT",
                "funding_rate": -0.00005,
                "premium_rate": -0.00005,
                "at_interest_floor": False,
                "passes_depth": True,
                "depth_usdt": 400_000,
                "side_hint": "long_perp_short_spot",
            },
            {
                "symbol": "BTC/USDT:USDT",
                "funding_rate": -0.0002,
                "premium_rate": -0.0003,
                "at_interest_floor": False,
                "passes_depth": True,
                "depth_usdt": 200_000,
                "side_hint": "long_perp_short_spot",
            },
        ]
    )
    d = decide_rotation(pos, screen, min_edge_multiple=3.0)
    assert d.action == RotationAction.ROTATE
    assert d.ratio >= 3.0
    assert d.candidate["symbol"] == "BTC/USDT:USDT"


def test_rotation_skips_when_only_floor_candidates() -> None:
    pos = _pos()
    screen = _screen(
        [
            {
                "symbol": "ETH/USDT:USDT",
                "funding_rate": 0.0001,
                "premium_rate": 0.0,
                "at_interest_floor": True,
                "passes_depth": True,
                "depth_usdt": 300_000,
                "side_hint": "short_perp_long_spot",
            }
        ]
    )
    d = decide_rotation(pos, screen, min_edge_multiple=3.0, require_non_floor=True)
    assert d.action == RotationAction.SKIP


def test_exit_on_adverse_funding_for_short_perp() -> None:
    pos = _pos("XRP/USDT:USDT", side="short_perp_long_spot", premium=0.00005)
    screen = _screen(
        [
            {
                "symbol": "XRP/USDT:USDT",
                "funding_rate": -0.00002,
                "premium_rate": -0.00012,
                "at_interest_floor": False,
                "passes_depth": True,
                "depth_usdt": 100_000,
                "side_hint": "long_perp_short_spot",
            }
        ]
    )
    ex = decide_exit(pos, screen)
    assert ex.action == ExitAction.EXIT
    assert ex.reason == ExitReason.ADVERSE_FUNDING


def test_exit_on_interest_floor() -> None:
    pos = _pos("ETH/USDT:USDT", side="short_perp_long_spot", premium=0.0001)
    screen = _screen(
        [
            {
                "symbol": "ETH/USDT:USDT",
                "funding_rate": 0.0001,
                "premium_rate": 0.0,
                "at_interest_floor": True,
                "passes_depth": True,
                "depth_usdt": 100_000,
                "side_hint": "short_perp_long_spot",
            }
        ]
    )
    ex = decide_exit(pos, screen)
    assert ex.action == ExitAction.EXIT
    assert ex.reason == ExitReason.INTEREST_FLOOR


def test_control_exit_before_rotation_same_cycle() -> None:
    """Juicy non-floor candidate exists, but floor exit wins and blocks rotate."""
    pos = _pos("ETH/USDT:USDT", side="short_perp_long_spot", premium=0.0001)
    screen = _screen(
        [
            {
                "symbol": "ETH/USDT:USDT",
                "funding_rate": 0.0001,
                "premium_rate": 0.0,
                "at_interest_floor": True,
                "passes_depth": True,
                "depth_usdt": 100_000,
                "side_hint": "short_perp_long_spot",
            },
            {
                "symbol": "BTC/USDT:USDT",
                "funding_rate": -0.0005,
                "premium_rate": -0.0006,
                "at_interest_floor": False,
                "passes_depth": True,
                "depth_usdt": 200_000,
                "side_hint": "long_perp_short_spot",
            },
        ]
    )
    result = run_control_cycle(screen, pos)
    assert result.action == CycleAction.EXIT
    assert result.rotation_decision is None
    assert any("rotation_skipped_after_exit" in n for n in result.notes)


def test_control_rotate_when_no_exit() -> None:
    pos = _pos(premium=-0.00005)
    screen = _screen(
        [
            {
                "symbol": "SOL/USDT:USDT",
                "funding_rate": -0.00005,
                "premium_rate": -0.00005,
                "at_interest_floor": False,
                "passes_depth": True,
                "depth_usdt": 400_000,
                "side_hint": "long_perp_short_spot",
            },
            {
                "symbol": "BTC/USDT:USDT",
                "funding_rate": -0.0002,
                "premium_rate": -0.0003,
                "at_interest_floor": False,
                "passes_depth": True,
                "depth_usdt": 200_000,
                "side_hint": "long_perp_short_spot",
            },
        ]
    )
    result = run_control_cycle(screen, pos)
    assert result.action == CycleAction.ROTATE
    assert result.position_after is not None
    assert result.position_after.symbol == "BTC/USDT:USDT"


def test_control_exit_unpaired_close_flattens() -> None:
    pos = _pos("ETH/USDT:USDT", side="short_perp_long_spot", premium=0.0001)
    screen = _screen(
        [
            {
                "symbol": "ETH/USDT:USDT",
                "funding_rate": 0.0001,
                "premium_rate": 0.0,
                "at_interest_floor": True,
                "passes_depth": True,
                "depth_usdt": 100_000,
                "side_hint": "short_perp_long_spot",
            }
        ]
    )
    result = run_control_cycle(screen, pos, close_perp_fill_ratio=0.0)
    assert result.action == CycleAction.FLATTEN_ALARM
    assert result.exec_result is not None
    assert result.exec_result.state.value == "flattened"
    assert result.position_after is None
