"""Step 6 — trade journal with separate gross vs net PnL columns."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from funding_arb.control import CycleAction, CycleResult
from funding_arb.position import PaperPosition

Reason = Literal[
    "entry",
    "exit_adverse",
    "exit_floor",
    "exit_other",
    "rotate",
    "flatten_alarm",
    "hold",
]

JOURNAL_COLUMNS = [
    "ts_utc",
    "exchange_id",
    "symbol",
    "side_hint",
    "reason",
    "funding_rate",
    "premium_rate",
    "at_interest_floor",
    "notional_usdt",
    "hold_hours",
    "est_settlements",
    "gross_pnl_usdt",
    "rotation_cost_usdt",
    "slippage_cost_usdt",
    "net_pnl_usdt",
    "gross_return",
    "net_return",
    "notes",
]


@dataclass
class JournalCosts:
    """Assumed frictional costs for paper gross→net (not live fills)."""

    rotation_cost: float = 0.004  # fraction of notional on rotate
    slippage_cost: float = 0.0005  # fraction per open or close event


@dataclass
class JournalRow:
    ts_utc: str
    exchange_id: str
    symbol: str
    side_hint: str
    reason: Reason
    funding_rate: float | None
    premium_rate: float | None
    at_interest_floor: bool | None
    notional_usdt: float
    hold_hours: float
    est_settlements: float
    gross_pnl_usdt: float
    rotation_cost_usdt: float
    slippage_cost_usdt: float
    net_pnl_usdt: float
    gross_return: float
    net_return: float
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def journal_dir(root: Path | str, exchange_id: str) -> Path:
    return Path(root) / exchange_id / "funding_arb" / "journal"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def estimate_hold_hours(opened_at: str | None, closed_at: str | None = None) -> float:
    start = _parse_ts(opened_at)
    end = _parse_ts(closed_at) or datetime.now(timezone.utc)
    if start is None:
        return 0.0
    return max(0.0, (end - start).total_seconds() / 3600.0)


def estimate_settlements(hold_hours: float, *, settlements_per_day: float = 3.0) -> float:
    return float(hold_hours) * float(settlements_per_day) / 24.0


def estimate_gross_funding_pnl(
    *,
    notional_usdt: float,
    side_hint: str,
    entry_funding_rate: float,
    exit_funding_rate: float | None,
    hold_hours: float,
    settlements_per_day: float = 3.0,
) -> float:
    """Paper gross ≈ notional × signed_avg_rate × settlements elapsed."""
    rates = [float(entry_funding_rate)]
    if exit_funding_rate is not None:
        rates.append(float(exit_funding_rate))
    avg_rate = sum(rates) / len(rates)
    settlements = estimate_settlements(hold_hours, settlements_per_day=settlements_per_day)
    if side_hint == "short_perp_long_spot":
        signed = avg_rate
    elif side_hint == "long_perp_short_spot":
        signed = -avg_rate
    else:
        signed = 0.0
    return float(notional_usdt) * signed * settlements


def append_journal_row(
    row: JournalRow | dict[str, Any],
    *,
    root: Path | str,
    exchange_id: str,
) -> list[Path]:
    data = row.to_dict() if isinstance(row, JournalRow) else dict(row)
    base = journal_dir(root, exchange_id)
    base.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([data], columns=JOURNAL_COLUMNS)
    written: list[Path] = []

    csv_path = base / "trades.csv"
    if csv_path.is_file():
        frame.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        frame.to_csv(csv_path, index=False)
    written.append(csv_path)

    pq_path = base / "trades.parquet"
    if pq_path.is_file():
        prev = pd.read_parquet(pq_path)
        out = pd.concat([prev, frame], ignore_index=True)
    else:
        out = frame
    out.to_parquet(pq_path, index=False)
    written.append(pq_path)

    meta = base / "trades.meta.json"
    meta.write_text(
        json.dumps(
            {"rows": int(len(out)), "updated_at": _utc_now(), "columns": list(out.columns)},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    written.append(meta)
    return written


def load_journal(root: Path | str, exchange_id: str) -> pd.DataFrame:
    path = journal_dir(root, exchange_id) / "trades.parquet"
    if path.is_file():
        return pd.read_parquet(path)
    csv_path = journal_dir(root, exchange_id) / "trades.csv"
    if csv_path.is_file():
        return pd.read_csv(csv_path)
    return pd.DataFrame(columns=JOURNAL_COLUMNS)


def reason_from_cycle(result: CycleResult) -> Reason:
    if result.action == CycleAction.ENTER:
        return "entry"
    if result.action == CycleAction.ROTATE:
        return "rotate"
    if result.action == CycleAction.FLATTEN_ALARM:
        return "flatten_alarm"
    if result.action == CycleAction.EXIT:
        raw = result.exit_decision.reason.value if result.exit_decision else ""
        if raw == "adverse_funding":
            return "exit_adverse"
        if raw == "interest_floor":
            return "exit_floor"
        return "exit_other"
    return "hold"


def _mark_rates(result: CycleResult, pos: PaperPosition) -> tuple[float | None, float | None, bool | None]:
    mark: dict[str, Any] | None = None
    if result.exit_decision and result.exit_decision.mark:
        mark = result.exit_decision.mark
    elif result.rotation_decision and result.rotation_decision.candidate:
        mark = result.rotation_decision.candidate
    if mark is None:
        return pos.entry_funding_rate, pos.entry_premium_rate, pos.entry_at_floor
    try:
        fr = float(mark["funding_rate"]) if mark.get("funding_rate") is not None else None
    except (TypeError, ValueError):
        fr = None
    try:
        pr = float(mark["premium_rate"]) if mark.get("premium_rate") is not None else None
    except (TypeError, ValueError):
        pr = None
    fl = mark.get("at_interest_floor")
    return fr, pr, bool(fl) if fl is not None else None


def journal_rows_from_cycle(
    result: CycleResult,
    *,
    costs: JournalCosts | None = None,
    settlements_per_day: float = 3.0,
    exchange_id: str = "okx",
) -> list[JournalRow]:
    """Build 0–2 journal rows for a control-cycle result (skip pure holds)."""
    costs = costs or JournalCosts()
    reason = reason_from_cycle(result)
    if reason == "hold":
        return []

    rows: list[JournalRow] = []
    ts = _utc_now()

    if result.position_before is not None and reason in {
        "exit_adverse",
        "exit_floor",
        "exit_other",
        "rotate",
        "flatten_alarm",
    }:
        before = result.position_before
        hold_h = estimate_hold_hours(before.opened_at, ts)
        exit_fr, exit_pr, exit_floor = _mark_rates(result, before)
        gross = estimate_gross_funding_pnl(
            notional_usdt=before.notional_usdt,
            side_hint=before.side_hint,
            entry_funding_rate=before.entry_funding_rate,
            exit_funding_rate=exit_fr,
            hold_hours=hold_h,
            settlements_per_day=settlements_per_day,
        )
        close_slip = float(before.notional_usdt) * float(costs.slippage_cost)
        rot_cost = (
            float(before.notional_usdt) * float(costs.rotation_cost)
            if reason == "rotate"
            else 0.0
        )
        net = gross - close_slip - rot_cost
        rows.append(
            JournalRow(
                ts_utc=ts,
                exchange_id=exchange_id or before.exchange_id,
                symbol=before.symbol,
                side_hint=before.side_hint,
                reason=reason,
                funding_rate=exit_fr,
                premium_rate=exit_pr,
                at_interest_floor=exit_floor,
                notional_usdt=float(before.notional_usdt),
                hold_hours=hold_h,
                est_settlements=estimate_settlements(
                    hold_h, settlements_per_day=settlements_per_day
                ),
                gross_pnl_usdt=gross,
                rotation_cost_usdt=rot_cost,
                slippage_cost_usdt=close_slip,
                net_pnl_usdt=net,
                gross_return=gross / before.notional_usdt if before.notional_usdt else 0.0,
                net_return=net / before.notional_usdt if before.notional_usdt else 0.0,
                notes=";".join(result.notes),
            )
        )

    if result.position_after is not None and reason in {"entry", "rotate"}:
        after = result.position_after
        open_slip = float(after.notional_usdt) * float(costs.slippage_cost)
        rows.append(
            JournalRow(
                ts_utc=ts,
                exchange_id=exchange_id or after.exchange_id,
                symbol=after.symbol,
                side_hint=after.side_hint,
                reason="entry",
                funding_rate=float(after.entry_funding_rate),
                premium_rate=float(after.entry_premium_rate),
                at_interest_floor=bool(after.entry_at_floor),
                notional_usdt=float(after.notional_usdt),
                hold_hours=0.0,
                est_settlements=0.0,
                gross_pnl_usdt=0.0,
                rotation_cost_usdt=0.0,
                slippage_cost_usdt=open_slip,
                net_pnl_usdt=-open_slip,
                gross_return=0.0,
                net_return=-open_slip / after.notional_usdt if after.notional_usdt else 0.0,
                notes=("rotate_open;" if reason == "rotate" else "") + ";".join(result.notes),
            )
        )

    return rows


def record_cycle(
    result: CycleResult,
    *,
    root: Path | str,
    exchange_id: str,
    costs: JournalCosts | None = None,
    settlements_per_day: float = 3.0,
) -> list[Path]:
    """Persist journal rows for a cycle; no-op on pure hold."""
    rows = journal_rows_from_cycle(
        result,
        costs=costs,
        settlements_per_day=settlements_per_day,
        exchange_id=exchange_id,
    )
    written: list[Path] = []
    for row in rows:
        written = append_journal_row(row, root=root, exchange_id=exchange_id)
    return written
