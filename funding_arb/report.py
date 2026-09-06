"""Step 6 — multi-day observation report (history + journal)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from funding_arb.journal import journal_dir, load_journal
from funding_arb.screen import compute_floor_persistence, load_history
from funding_arb.storage import snapshots_dir


def _parse_ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def observation_readiness(
    history: pd.DataFrame,
    *,
    min_days: float = 5.0,
    min_settlements: int = 15,
    settlements_per_day: float = 3.0,
    time_col: str = "observed_at",
) -> dict[str, Any]:
    """True only after enough wall-clock history (not after coding finishes)."""
    if history.empty or time_col not in history.columns:
        return {
            "ready": False,
            "span_days": 0.0,
            "n_snapshot_times": 0,
            "est_settlements": 0.0,
            "min_days": min_days,
            "min_settlements": min_settlements,
            "reason": "empty_history",
        }
    ts = _parse_ts(history[time_col]).dropna()
    if ts.empty:
        return {
            "ready": False,
            "span_days": 0.0,
            "n_snapshot_times": int(len(history)),
            "est_settlements": 0.0,
            "min_days": min_days,
            "min_settlements": min_settlements,
            "reason": "unparseable_timestamps",
        }
    span_days = (ts.max() - ts.min()).total_seconds() / 86400.0
    n_times = int(ts.dt.floor("min").nunique())
    est_settlements = span_days * float(settlements_per_day)
    ready = span_days >= float(min_days) and est_settlements >= float(min_settlements)
    return {
        "ready": bool(ready),
        "span_days": float(span_days),
        "n_snapshot_times": n_times,
        "n_rows": int(len(history)),
        "est_settlements": float(est_settlements),
        "min_days": min_days,
        "min_settlements": min_settlements,
        "first_ts": ts.min().isoformat(),
        "last_ts": ts.max().isoformat(),
        "reason": "ok" if ready else "need_more_wall_clock_history",
    }


def history_premium_stats(history: pd.DataFrame) -> dict[str, Any]:
    if history.empty:
        return {"rows": 0}
    out: dict[str, Any] = {"rows": int(len(history))}
    if "at_interest_floor" in history.columns:
        out["floor_fraction"] = float(history["at_interest_floor"].fillna(False).mean())
    if "funding_rate" in history.columns and "interest_rate" in history.columns:
        prem = pd.to_numeric(history["funding_rate"], errors="coerce") - pd.to_numeric(
            history["interest_rate"], errors="coerce"
        )
        out["mean_abs_premium"] = float(prem.abs().mean())
        out["median_abs_premium"] = float(prem.abs().median())
    elif "premium_rate" in history.columns:
        prem = pd.to_numeric(history["premium_rate"], errors="coerce")
        out["mean_abs_premium"] = float(prem.abs().mean())
        out["median_abs_premium"] = float(prem.abs().median())
    return out


def journal_pnl_summary(journal: pd.DataFrame) -> dict[str, Any]:
    if journal.empty:
        return {
            "rows": 0,
            "sum_gross_pnl_usdt": 0.0,
            "sum_net_pnl_usdt": 0.0,
            "sum_rotation_cost_usdt": 0.0,
            "sum_slippage_cost_usdt": 0.0,
            "gross_minus_net_usdt": 0.0,
        }
    g = float(pd.to_numeric(journal["gross_pnl_usdt"], errors="coerce").fillna(0).sum())
    n = float(pd.to_numeric(journal["net_pnl_usdt"], errors="coerce").fillna(0).sum())
    rc = float(pd.to_numeric(journal["rotation_cost_usdt"], errors="coerce").fillna(0).sum())
    sc = float(pd.to_numeric(journal["slippage_cost_usdt"], errors="coerce").fillna(0).sum())
    by_reason: dict[str, float] = {}
    if "reason" in journal.columns:
        tmp = journal.copy()
        tmp["net_pnl_usdt"] = pd.to_numeric(tmp["net_pnl_usdt"], errors="coerce").fillna(0.0)
        by_reason = tmp.groupby("reason")["net_pnl_usdt"].sum().astype(float).to_dict()
    return {
        "rows": int(len(journal)),
        "sum_gross_pnl_usdt": g,
        "sum_net_pnl_usdt": n,
        "sum_rotation_cost_usdt": rc,
        "sum_slippage_cost_usdt": sc,
        "gross_minus_net_usdt": g - n,
        "by_reason": by_reason,
    }


def build_observation_report(
    *,
    root: Path | str,
    exchange_id: str,
    floor_lookback: int = 48,
    min_days: float = 5.0,
    min_settlements: int = 15,
    settlements_per_day: float = 3.0,
) -> dict[str, Any]:
    history = load_history(root, exchange_id)
    journal = load_journal(root, exchange_id)
    readiness = observation_readiness(
        history,
        min_days=min_days,
        min_settlements=min_settlements,
        settlements_per_day=settlements_per_day,
    )
    persistence = (
        compute_floor_persistence(history, lookback_snapshots=floor_lookback)
        if not history.empty
        else pd.DataFrame()
    )
    persist_summary: dict[str, Any] = {"symbols": 0}
    if not persistence.empty and "floor_persistence" in persistence.columns:
        persist_summary = {
            "symbols": int(len(persistence)),
            "mean_floor_persistence": float(persistence["floor_persistence"].mean()),
            "median_floor_persistence": float(persistence["floor_persistence"].median()),
            "always_floor_count": int((persistence["floor_persistence"] >= 0.99).sum()),
            "never_floor_count": int((persistence["floor_persistence"] <= 0.01).sum()),
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "exchange_id": exchange_id,
        "readiness": readiness,
        "history_premium": history_premium_stats(history),
        "floor_persistence": persist_summary,
        "journal_pnl": journal_pnl_summary(journal),
        "paths": {
            "history": str(snapshots_dir(root, exchange_id) / "history.parquet"),
            "journal": str(journal_dir(root, exchange_id) / "trades.csv"),
        },
        "note": (
            "READY for Module-1 readiness review"
            if readiness.get("ready")
            else "NOT READY — keep collect --loop running; do not start live capital yet"
        ),
    }


def save_observation_report(
    report: dict[str, Any],
    *,
    root: Path | str,
    exchange_id: str,
) -> Path:
    base = Path(root) / exchange_id / "funding_arb" / "reports"
    base.mkdir(parents=True, exist_ok=True)
    path = base / "observation_latest.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    dated = base / f"observation_{datetime.now(timezone.utc).date().isoformat()}.json"
    dated.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path
