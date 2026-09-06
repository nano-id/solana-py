"""Persist live funding_arb snapshots (parquet + CSV latest table)."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from pairs_trading.data.storage import symbol_to_filename

PersistMode = Literal["dated", "latest", "both"]


def snapshots_dir(root: Path | str, exchange_id: str) -> Path:
    return Path(root) / exchange_id / "funding_arb" / "snapshots"


def append_history(frame: pd.DataFrame, path: Path) -> Path:
    """Append rows to a cumulative parquet history file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        prev = pd.read_parquet(path)
        out = pd.concat([prev, frame], ignore_index=True)
    else:
        out = frame.copy()
    out.to_parquet(path, index=False)
    return path


def write_latest_table(frame: pd.DataFrame, directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    parquet = directory / "latest.parquet"
    csv = directory / "latest.csv"
    meta = directory / "latest.meta.json"
    frame.to_parquet(parquet, index=False)
    frame.to_csv(csv, index=False)
    meta.write_text(
        json.dumps(
            {
                "rows": int(len(frame)),
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "columns": list(frame.columns),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    paths.extend([parquet, csv, meta])
    return paths


def save_snapshot_batch(
    frame: pd.DataFrame,
    *,
    root: Path | str,
    exchange_id: str,
    mode: PersistMode = "both",
) -> list[Path]:
    """Write latest table and/or dated + rolling history."""
    if frame.empty:
        return []
    base = snapshots_dir(root, exchange_id)
    written: list[Path] = []
    today = date.today().isoformat()

    if mode in ("latest", "both"):
        written.extend(write_latest_table(frame, base))

    if mode in ("dated", "both"):
        dated_csv = base / f"snapshot_{today}.csv"
        dated_parquet = base / f"snapshot_{today}.parquet"
        # Append-friendly: concat if same-day file already exists.
        if dated_parquet.is_file():
            prev = pd.read_parquet(dated_parquet)
            day = pd.concat([prev, frame], ignore_index=True)
        else:
            day = frame.copy()
        base.mkdir(parents=True, exist_ok=True)
        day.to_parquet(dated_parquet, index=False)
        day.to_csv(dated_csv, index=False)
        written.extend([dated_parquet, dated_csv])

        hist = append_history(frame, base / "history.parquet")
        written.append(hist)

    return written


def save_per_symbol_ticks(
    frame: pd.DataFrame,
    *,
    root: Path | str,
    exchange_id: str,
) -> list[Path]:
    """Optional per-symbol tick append (one parquet per coin)."""
    if frame.empty or "symbol" not in frame.columns:
        return []
    out_dir = Path(root) / exchange_id / "funding_arb" / "ticks"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for sym, grp in frame.groupby("symbol", sort=False):
        path = out_dir / f"{symbol_to_filename(str(sym))}.parquet"
        append_history(grp.reset_index(drop=True), path)
        paths.append(path)
    return paths


def snapshot_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"rows": 0}
    fr = frame["funding_rate"].dropna() if "funding_rate" in frame.columns else pd.Series(dtype=float)
    depth = frame["depth_usdt"].dropna() if "depth_usdt" in frame.columns else pd.Series(dtype=float)
    return {
        "rows": int(len(frame)),
        "with_funding": int(fr.shape[0]),
        "with_depth": int(depth.shape[0]),
        "median_abs_funding": float(fr.abs().median()) if not fr.empty else None,
        "median_depth_usdt": float(depth.median()) if not depth.empty else None,
    }
