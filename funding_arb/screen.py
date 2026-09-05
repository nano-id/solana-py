"""Step 2 — opportunity screen: annualized funding rank + depth filter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pandas as pd

from funding_arb.storage import snapshots_dir

PersistMode = Literal["dated", "latest", "both"]


def annualized_funding(
    funding_rate: float,
    *,
    settlements_per_day: float = 3.0,
) -> float:
    """Convert per-settlement funding rate to simple annualized return.

    8h cycle → 3 settlements/day → rate × 3 × 365.
    """
    return float(funding_rate) * float(settlements_per_day) * 365.0


def screen_opportunities(
    snapshot: pd.DataFrame,
    *,
    settlements_per_day: float = 3.0,
    min_depth_usdt: float = 10_000.0,
    rate_col: str = "funding_rate",
) -> pd.DataFrame:
    """Rank symbols by |annualized funding|; drop thin books.

    Uses ``funding_rate`` (next settlement) for the primary score. When many
    names sit on the OKX interest floor (0.0001), ``sett_funding_rate`` and
    ``at_interest_floor`` are kept as tie-break / honesty columns — they do
    not replace the primary rate (that is what you earn next).
    """
    if snapshot.empty:
        return pd.DataFrame()

    frame = snapshot.copy()
    if rate_col not in frame.columns:
        raise KeyError(f"snapshot missing rate column {rate_col!r}")

    rates = pd.to_numeric(frame[rate_col], errors="coerce")
    frame["funding_rate"] = rates
    frame["ann_funding"] = rates.map(
        lambda r: annualized_funding(r, settlements_per_day=settlements_per_day)
        if pd.notna(r)
        else float("nan")
    )
    frame["ann_funding_abs"] = frame["ann_funding"].abs()

    if "sett_funding_rate" in frame.columns:
        sett = pd.to_numeric(frame["sett_funding_rate"], errors="coerce")
        frame["ann_sett_funding"] = sett.map(
            lambda r: annualized_funding(r, settlements_per_day=settlements_per_day)
            if pd.notna(r)
            else float("nan")
        )
    else:
        frame["ann_sett_funding"] = float("nan")

    depth = (
        pd.to_numeric(frame["depth_usdt"], errors="coerce")
        if "depth_usdt" in frame.columns
        else pd.Series([float("nan")] * len(frame), index=frame.index)
    )
    frame["depth_usdt"] = depth
    frame["passes_depth"] = depth >= float(min_depth_usdt)
    frame["side_hint"] = rates.map(
        lambda r: "short_perp_long_spot"
        if pd.notna(r) and r > 0
        else ("long_perp_short_spot" if pd.notna(r) and r < 0 else "flat")
    )

    # Eligible first, then by |ann|; floor-tied names break on |sett| then depth.
    frame = frame.sort_values(
        by=["passes_depth", "ann_funding_abs", "ann_sett_funding", "depth_usdt"],
        ascending=[False, False, False, False],
        key=lambda s: s.abs() if s.name in {"ann_sett_funding"} else s,
    ).reset_index(drop=True)
    # Fix sett sort: we want abs for tie-break — redo explicitly.
    frame["_sett_abs"] = frame["ann_sett_funding"].abs()
    frame = frame.sort_values(
        by=["passes_depth", "ann_funding_abs", "_sett_abs", "depth_usdt"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    frame = frame.drop(columns=["_sett_abs"])
    frame.insert(0, "rank", range(1, len(frame) + 1))
    return frame


def screen_summary(screen: pd.DataFrame) -> dict[str, Any]:
    if screen.empty:
        return {"rows": 0, "eligible": 0, "at_interest_floor": 0}
    eligible = screen[screen["passes_depth"]] if "passes_depth" in screen.columns else screen
    at_floor = (
        int(screen["at_interest_floor"].fillna(False).astype(bool).sum())
        if "at_interest_floor" in screen.columns
        else 0
    )
    return {
        "rows": int(len(screen)),
        "eligible": int(len(eligible)),
        "at_interest_floor": at_floor,
        "top_symbol": str(eligible.iloc[0]["symbol"]) if len(eligible) else None,
        "top_ann_funding": float(eligible.iloc[0]["ann_funding"]) if len(eligible) else None,
        "min_depth_usdt_eligible": float(eligible["depth_usdt"].min())
        if len(eligible) and "depth_usdt" in eligible.columns
        else None,
    }


def save_screen_batch(
    screen: pd.DataFrame,
    *,
    root: Path | str,
    exchange_id: str,
    mode: PersistMode = "both",
    min_depth_usdt: float | None = None,
) -> list[Path]:
    """Persist screen table under ``.../funding_arb/screens/``."""
    if screen.empty:
        return []
    base = Path(root) / exchange_id / "funding_arb" / "screens"
    base.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def _write(stem: str) -> None:
        pq = base / f"{stem}.parquet"
        csv = base / f"{stem}.csv"
        screen.to_parquet(pq, index=False)
        screen.to_csv(csv, index=False)
        written.extend([pq, csv])

    if mode in ("latest", "both"):
        _write("latest")
        meta = base / "latest.meta.json"
        import json
        from datetime import datetime, timezone

        meta.write_text(
            json.dumps(
                {
                    **screen_summary(screen),
                    "min_depth_usdt": min_depth_usdt,
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        written.append(meta)

    if mode in ("dated", "both"):
        from datetime import date

        _write(f"screen_{date.today().isoformat()}")

    return written


# Re-export for callers that already import snapshots_dir.
__all__ = [
    "annualized_funding",
    "screen_opportunities",
    "screen_summary",
    "save_screen_batch",
    "snapshots_dir",
]
