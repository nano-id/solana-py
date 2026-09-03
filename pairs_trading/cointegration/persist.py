"""Persist cointegration scan outputs (dated and/or latest)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd


def scan_output_dir(root: Path, exchange_id: str) -> Path:
    return Path(root) / exchange_id / "cointegration"


def save_scan_results(
    df: pd.DataFrame,
    *,
    root: Path,
    exchange_id: str,
    timeframe: str,
    mode: Literal["dated", "latest", "both"] = "both",
    meta: dict[str, Any] | None = None,
    as_of: datetime | None = None,
) -> dict[str, Path]:
    """Write results parquet (+ meta json). Returns written paths."""
    as_of = as_of or datetime.now(timezone.utc)
    base = scan_output_dir(root, exchange_id)
    base.mkdir(parents=True, exist_ok=True)

    stamp = as_of.astimezone(timezone.utc).strftime("%Y-%m-%d")
    written: dict[str, Path] = {}

    targets: list[tuple[str, Path]] = []
    if mode in ("dated", "both"):
        targets.append(("dated", base / f"scan_{stamp}_{timeframe}.parquet"))
    if mode in ("latest", "both"):
        targets.append(("latest", base / f"scan_latest_{timeframe}.parquet"))

    payload_meta = {
        "as_of": as_of.astimezone(timezone.utc).isoformat(),
        "exchange_id": exchange_id,
        "timeframe": timeframe,
        "n_hits": int(len(df)),
        **(meta or {}),
    }

    for label, path in targets:
        df.to_parquet(path, index=False)
        meta_path = path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(payload_meta, indent=2), encoding="utf-8")
        written[label] = path
        written[f"{label}_meta"] = meta_path

    return written
