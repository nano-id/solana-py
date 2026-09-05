"""Persist harvest / screen artifacts under data/{exchange}/funding/screens/."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd


def save_funding_screen(
    df: pd.DataFrame,
    *,
    root: Path,
    exchange_id: str,
    name: str = "harvest",
    mode: Literal["dated", "latest", "both"] = "both",
    meta: dict[str, Any] | None = None,
) -> list[Path]:
    base = Path(root) / exchange_id / "funding" / "screens"
    base.mkdir(parents=True, exist_ok=True)
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    written: list[Path] = []
    targets: list[Path] = []
    if mode in ("dated", "both"):
        targets.append(base / f"{name}_{day}")
    if mode in ("latest", "both"):
        targets.append(base / f"{name}_latest")

    payload_meta = {
        "name": name,
        "exchange_id": exchange_id,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(df)),
        **(meta or {}),
    }
    for stem in targets:
        pq = Path(str(stem) + ".parquet")
        csv = Path(str(stem) + ".csv")
        meta_path = Path(str(stem) + ".meta.json")
        df.to_parquet(pq, index=False)
        df.to_csv(csv, index=False)
        meta_path.write_text(json.dumps(payload_meta, indent=2), encoding="utf-8")
        written.extend([pq, csv, meta_path])
    return written
