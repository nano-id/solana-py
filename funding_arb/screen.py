"""Step 2 — opportunity screen: annualized funding + depth + floor persistence."""

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


def compute_floor_persistence(
    history: pd.DataFrame,
    *,
    lookback_snapshots: int = 12,
    symbol_col: str = "symbol",
    floor_col: str = "at_interest_floor",
    time_col: str = "observed_at",
) -> pd.DataFrame:
    """Fraction of the last N snapshots each symbol spent at the interest floor.

    Also reports mean |funding − interest| and mean |sett − interest| over the
    window so we can see whether floor names ever print real premium.
    """
    empty = pd.DataFrame(
        columns=[
            "symbol",
            "floor_persistence",
            "floor_snapshots",
            "lookback_used",
            "mean_abs_premium",
            "mean_abs_sett_premium",
        ]
    )
    if history.empty or symbol_col not in history.columns:
        return empty

    frame = history.copy()
    if time_col in frame.columns:
        frame = frame.sort_values(time_col)
    if floor_col not in frame.columns:
        frame[floor_col] = False

    rows: list[dict[str, Any]] = []
    for sym, grp in frame.groupby(symbol_col, sort=False):
        window = grp.tail(int(lookback_snapshots))
        n = int(len(window))
        floors = window[floor_col].fillna(False).astype(bool)
        n_floor = int(floors.sum())
        persistence = float(n_floor / n) if n else float("nan")

        mean_abs_premium = float("nan")
        mean_abs_sett = float("nan")
        if "funding_rate" in window.columns and "interest_rate" in window.columns:
            prem = (
                pd.to_numeric(window["funding_rate"], errors="coerce")
                - pd.to_numeric(window["interest_rate"], errors="coerce")
            )
            mean_abs_premium = float(prem.abs().mean())
        if "sett_funding_rate" in window.columns and "interest_rate" in window.columns:
            sett_p = (
                pd.to_numeric(window["sett_funding_rate"], errors="coerce")
                - pd.to_numeric(window["interest_rate"], errors="coerce")
            )
            mean_abs_sett = float(sett_p.abs().mean())

        rows.append(
            {
                "symbol": sym,
                "floor_persistence": persistence,
                "floor_snapshots": n_floor,
                "lookback_used": n,
                "mean_abs_premium": mean_abs_premium,
                "mean_abs_sett_premium": mean_abs_sett,
            }
        )
    return pd.DataFrame(rows)


def load_history(root: Path | str, exchange_id: str) -> pd.DataFrame:
    path = snapshots_dir(root, exchange_id) / "history.parquet"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_parquet(path)


def screen_opportunities(
    snapshot: pd.DataFrame,
    *,
    settlements_per_day: float = 3.0,
    min_depth_usdt: float = 10_000.0,
    rate_col: str = "funding_rate",
    history: pd.DataFrame | None = None,
    floor_lookback: int = 12,
    prefer_non_floor: bool = True,
) -> pd.DataFrame:
    """Rank symbols for funding harvest.

    Priority (when ``prefer_non_floor``):
      1. passes depth filter
      2. low ``floor_persistence`` / not currently at interest floor
         (real premium over OKX structural interest)
      3. |premium| = |funding − interest|
      4. |ann_funding|, then depth

    Floor-parked names stay visible but rank below true-premium names so
    entry logic does not mistake the interest base rate for edge.
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

    interest = (
        pd.to_numeric(frame["interest_rate"], errors="coerce")
        if "interest_rate" in frame.columns
        else pd.Series([float("nan")] * len(frame), index=frame.index)
    )
    frame["premium_rate"] = rates - interest
    frame["premium_abs"] = frame["premium_rate"].abs()
    frame["ann_premium"] = frame["premium_rate"].map(
        lambda r: annualized_funding(r, settlements_per_day=settlements_per_day)
        if pd.notna(r)
        else float("nan")
    )

    if "sett_funding_rate" in frame.columns:
        sett = pd.to_numeric(frame["sett_funding_rate"], errors="coerce")
        frame["ann_sett_funding"] = sett.map(
            lambda r: annualized_funding(r, settlements_per_day=settlements_per_day)
            if pd.notna(r)
            else float("nan")
        )
        frame["sett_premium_rate"] = sett - interest
    else:
        frame["ann_sett_funding"] = float("nan")
        frame["sett_premium_rate"] = float("nan")

    if "at_interest_floor" not in frame.columns:
        frame["at_interest_floor"] = False
    frame["at_interest_floor"] = frame["at_interest_floor"].fillna(False).astype(bool)

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

    # Floor persistence from history (defaults to current flag if no history).
    if history is not None and not history.empty:
        pers = compute_floor_persistence(
            history, lookback_snapshots=floor_lookback
        )
        frame = frame.merge(pers, on="symbol", how="left")
    else:
        frame["floor_persistence"] = frame["at_interest_floor"].astype(float)
        frame["floor_snapshots"] = frame["at_interest_floor"].astype(int)
        frame["lookback_used"] = 1
        frame["mean_abs_premium"] = frame["premium_abs"]
        frame["mean_abs_sett_premium"] = frame["sett_premium_rate"].abs()

    frame["floor_persistence"] = pd.to_numeric(
        frame["floor_persistence"], errors="coerce"
    ).fillna(frame["at_interest_floor"].astype(float))

    # Entry score: prefer non-floor real premium.
    frame["prefer_non_floor"] = (~frame["at_interest_floor"]).astype(int)
    if prefer_non_floor:
        frame["_sett_abs"] = frame["ann_sett_funding"].abs()
        frame = frame.sort_values(
            by=[
                "passes_depth",
                "prefer_non_floor",
                "floor_persistence",
                "premium_abs",
                "ann_funding_abs",
                "_sett_abs",
                "depth_usdt",
            ],
            ascending=[False, False, True, False, False, False, False],
        ).reset_index(drop=True)
        frame = frame.drop(columns=["_sett_abs"])
    else:
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
        return {"rows": 0, "eligible": 0, "at_interest_floor": 0, "eligible_non_floor": 0}
    eligible = (
        screen[screen["passes_depth"]] if "passes_depth" in screen.columns else screen
    )
    at_floor = (
        int(screen["at_interest_floor"].fillna(False).astype(bool).sum())
        if "at_interest_floor" in screen.columns
        else 0
    )
    eligible_non_floor = 0
    if not eligible.empty and "at_interest_floor" in eligible.columns:
        eligible_non_floor = int((~eligible["at_interest_floor"].astype(bool)).sum())
    top = eligible.iloc[0] if len(eligible) else None
    return {
        "rows": int(len(screen)),
        "eligible": int(len(eligible)),
        "at_interest_floor": at_floor,
        "eligible_non_floor": eligible_non_floor,
        "top_symbol": str(top["symbol"]) if top is not None else None,
        "top_ann_funding": float(top["ann_funding"]) if top is not None else None,
        "top_at_floor": bool(top["at_interest_floor"]) if top is not None else None,
        "top_floor_persistence": float(top["floor_persistence"])
        if top is not None and "floor_persistence" in eligible.columns
        else None,
        "top_premium_rate": float(top["premium_rate"])
        if top is not None and "premium_rate" in eligible.columns
        else None,
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


__all__ = [
    "annualized_funding",
    "compute_floor_persistence",
    "load_history",
    "screen_opportunities",
    "screen_summary",
    "save_screen_batch",
    "snapshots_dir",
]
