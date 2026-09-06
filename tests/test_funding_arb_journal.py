"""Tests for step 6 journal (gross vs net) and observation readiness."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from funding_arb.control import CycleAction, CycleResult
from funding_arb.exit_rules import ExitAction, ExitDecision, ExitReason
from funding_arb.journal import JournalCosts, load_journal, record_cycle
from funding_arb.position import PaperPosition
from funding_arb.report import build_observation_report, observation_readiness


def _pos(*, hours_open: float = 16.0) -> PaperPosition:
    opened = datetime.now(timezone.utc) - timedelta(hours=hours_open)
    return PaperPosition(
        symbol="SOL/USDT:USDT",
        side_hint="short_perp_long_spot",
        notional_usdt=100.0,
        entry_funding_rate=0.0002,
        entry_premium_rate=0.0001,
        entry_at_floor=False,
        opened_at=opened.isoformat(),
    )


def test_journal_exit_separates_gross_and_net(tmp_path: Path) -> None:
    pos = _pos(hours_open=16.0)
    # 16h ≈ 2 settlements; short_perp collects +0.0002 * 2 * 100 = 0.04 gross
    ex = ExitDecision(
        action=ExitAction.EXIT,
        reason=ExitReason.ADVERSE_FUNDING,
        position=pos,
        mark={
            "funding_rate": 0.0002,
            "premium_rate": 0.0001,
            "at_interest_floor": False,
        },
        funding_rate=0.0002,
        at_interest_floor=False,
        details=["adverse"],
    )
    result = CycleResult(
        action=CycleAction.EXIT,
        position_before=pos,
        position_after=None,
        exit_decision=ex,
        notes=["exit:adverse_funding"],
    )
    paths = record_cycle(
        result,
        root=tmp_path,
        exchange_id="okx",
        costs=JournalCosts(rotation_cost=0.004, slippage_cost=0.0005),
    )
    assert paths
    journal = load_journal(tmp_path, "okx")
    assert len(journal) == 1
    row = journal.iloc[0]
    assert row["reason"] == "exit_adverse"
    assert float(row["gross_pnl_usdt"]) > 0
    assert float(row["slippage_cost_usdt"]) > 0
    assert float(row["net_pnl_usdt"]) < float(row["gross_pnl_usdt"])
    assert abs(float(row["gross_pnl_usdt"]) - 0.04) < 1e-9


def test_journal_entry_books_open_slippage_only(tmp_path: Path) -> None:
    after = _pos(hours_open=0.0)
    result = CycleResult(
        action=CycleAction.ENTER,
        position_before=None,
        position_after=after,
        notes=["entered_non_floor"],
    )
    record_cycle(
        result,
        root=tmp_path,
        exchange_id="okx",
        costs=JournalCosts(slippage_cost=0.0005),
    )
    journal = load_journal(tmp_path, "okx")
    assert len(journal) == 1
    row = journal.iloc[0]
    assert row["reason"] == "entry"
    assert float(row["gross_pnl_usdt"]) == 0.0
    assert float(row["net_pnl_usdt"]) == -0.05  # 100 * 5bps


def test_observation_not_ready_on_short_history() -> None:
    now = datetime.now(timezone.utc)
    history = pd.DataFrame(
        [
            {
                "observed_at": (now - timedelta(hours=2)).isoformat(),
                "symbol": "BTC/USDT:USDT",
                "funding_rate": 0.0001,
                "interest_rate": 0.0001,
                "at_interest_floor": True,
            },
            {
                "observed_at": now.isoformat(),
                "symbol": "BTC/USDT:USDT",
                "funding_rate": 0.0001,
                "interest_rate": 0.0001,
                "at_interest_floor": True,
            },
        ]
    )
    ready = observation_readiness(history, min_days=5.0, min_settlements=15)
    assert ready["ready"] is False
    assert ready["span_days"] < 1.0


def test_observation_ready_after_enough_span() -> None:
    now = datetime.now(timezone.utc)
    history = pd.DataFrame(
        [
            {
                "observed_at": (now - timedelta(days=6)).isoformat(),
                "symbol": "BTC/USDT:USDT",
                "funding_rate": 0.0001,
                "at_interest_floor": True,
            },
            {
                "observed_at": now.isoformat(),
                "symbol": "BTC/USDT:USDT",
                "funding_rate": 0.00005,
                "at_interest_floor": False,
            },
        ]
    )
    ready = observation_readiness(history, min_days=5.0, min_settlements=15)
    assert ready["ready"] is True


def test_build_report_writes_not_ready_note(tmp_path: Path) -> None:
    # empty history → not ready
    report = build_observation_report(root=tmp_path, exchange_id="okx")
    assert report["readiness"]["ready"] is False
    assert "NOT READY" in report["note"]


def test_journal_pnl_summary_by_reason() -> None:
    from funding_arb.report import journal_pnl_summary

    journal = pd.DataFrame(
        [
            {
                "reason": "exit_adverse",
                "gross_pnl_usdt": 1.0,
                "net_pnl_usdt": 0.5,
                "rotation_cost_usdt": 0.0,
                "slippage_cost_usdt": 0.5,
            },
            {
                "reason": "rotate",
                "gross_pnl_usdt": 0.2,
                "net_pnl_usdt": -0.3,
                "rotation_cost_usdt": 0.4,
                "slippage_cost_usdt": 0.1,
            },
        ]
    )
    summary = journal_pnl_summary(journal)
    assert summary["rows"] == 2
    assert abs(summary["sum_gross_pnl_usdt"] - 1.2) < 1e-9
    assert abs(summary["gross_minus_net_usdt"] - 1.0) < 1e-9
    assert abs(summary["by_reason"]["exit_adverse"] - 0.5) < 1e-9
    assert abs(summary["by_reason"]["rotate"] - (-0.3)) < 1e-9
