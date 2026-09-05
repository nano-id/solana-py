"""Synthesize ~1/min BBO events from OHLC bars for event-driven paper MM.

Archived L2 is unavailable for most months; this builds a conservative
intra-bar mid path inside the OHLC envelope and emits BookSnapshot events.
PnL from this feed is indicative only (interpretation_allowed=False).
"""

from __future__ import annotations

import calendar
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from market_making.config import MarketMakingConfig
from market_making.feed import BookSnapshot
from market_making.realtime import RealtimeResult, run_realtime_paper, save_realtime_artifacts


def _month_bounds(year: int, month: int) -> tuple[str, str]:
    start = f"{year:04d}-{month:02d}-01"
    if month == 12:
        end = f"{year + 1:04d}-01-01"
    else:
        end = f"{year:04d}-{month + 1:02d}-01"
    return start, end


def load_ohlc_month(
    symbol: str,
    *,
    year: int,
    month: int,
    timeframe: str = "1h",
    root: str | Path = "data",
    exchange_id: str = "okx",
) -> pd.DataFrame:
    base = symbol.replace("/", "_")
    path = Path(root) / exchange_id / timeframe / f"{base}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    start, end = _month_bounds(year, month)
    mask = (ts >= pd.Timestamp(start, tz="UTC")) & (ts < pd.Timestamp(end, tz="UTC"))
    sl = df.loc[mask].sort_values("timestamp").copy()
    if len(sl) < 2:
        raise ValueError(f"Insufficient {timeframe} bars for {symbol} {start}: {len(sl)}")
    sl["datetime_utc"] = ts.loc[mask].sort_values().to_numpy()
    return sl


def synth_bbo_from_ohlc(
    df: pd.DataFrame,
    symbol: str,
    *,
    steps_per_bar: int = 60,
    seed: int = 42,
    exchange_id: str = "okx_hist_synth",
) -> list[BookSnapshot]:
    """Map each OHLC bar to ``steps_per_bar`` BBO snapshots (~1/min on 1h)."""
    rng = np.random.default_rng(seed + abs(hash(symbol)) % 10_000)
    out: list[BookSnapshot] = []
    bar_ms = int(
        (
            pd.Timestamp(df["timestamp"].iloc[1], unit="ms", tz="UTC")
            - pd.Timestamp(df["timestamp"].iloc[0], unit="ms", tz="UTC")
        ).total_seconds()
        * 1000
    )
    if bar_ms <= 0:
        bar_ms = 3_600_000
    step_ms = max(1, bar_ms // steps_per_bar)

    for _, row in df.iterrows():
        o, h, l, c = map(float, (row["open"], row["high"], row["low"], row["close"]))
        if min(o, h, l, c) <= 0:
            continue
        vol = float(row.get("volume", 0.0) or 0.0)
        path = np.linspace(o, c, steps_per_bar)
        amp = max((h - l) * 0.15, abs(c - o) * 0.05, c * 1e-5)
        noise = rng.normal(0.0, amp / 4.0, size=steps_per_bar)
        micro = np.clip(path + noise, l * 0.999, h * 1.001)
        half_bps = 1.5 + min(8.0, (h - l) / max(c, 1e-12) * 1e4 * 0.05)
        size = max(vol / (steps_per_bar * 40.0), 1e-8)
        t0 = int(row["timestamp"])
        for i, mid in enumerate(micro):
            mid = float(mid)
            hs = mid * half_bps / 1e4
            bid = mid - hs
            ask = mid + hs
            out.append(
                BookSnapshot(
                    symbol=symbol,
                    ts_ms=t0 + i * step_ms,
                    bid=bid,
                    ask=ask,
                    bid_size=size * (0.8 + 0.4 * rng.random()),
                    ask_size=size * (0.8 + 0.4 * rng.random()),
                    mid=mid,
                    spread_bps=float(2.0 * half_bps),
                    exchange_id=exchange_id,
                )
            )
    return out


def build_month_events(
    symbols: Iterable[str],
    *,
    year: int,
    month: int,
    timeframe: str = "1h",
    steps_per_bar: int = 60,
    seed: int = 42,
    root: str | Path = "data",
    exchange_id: str = "okx",
) -> list[BookSnapshot]:
    events: list[BookSnapshot] = []
    for sym in symbols:
        df = load_ohlc_month(
            sym,
            year=year,
            month=month,
            timeframe=timeframe,
            root=root,
            exchange_id=exchange_id,
        )
        events.extend(
            synth_bbo_from_ohlc(
                df, sym, steps_per_bar=steps_per_bar, seed=seed, exchange_id=f"{exchange_id}_hist_synth"
            )
        )
    events.sort(key=lambda e: (e.ts_ms, e.symbol))
    return events


def run_month_events_paper(
    cfg: MarketMakingConfig,
    *,
    year: int,
    month: int,
    timeframe: str = "1h",
    steps_per_bar: int = 60,
    seed: int = 42,
    initial_equity: float | None = None,
    fill_prob: float | None = None,
    adverse_frac: float | None = None,
) -> tuple[RealtimeResult, list[dict], dict]:
    """Event-driven paper over a calendar month using synth BBO from OHLC."""
    updates: dict = {}
    if fill_prob is not None or adverse_frac is not None:
        sim = cfg.sim.model_copy()
        if fill_prob is not None:
            sim.fill_prob = float(fill_prob)
        if adverse_frac is not None:
            sim.adverse_frac = float(adverse_frac)
        updates["sim"] = sim
    run_cfg = cfg.model_copy(update=updates) if updates else cfg

    events = build_month_events(
        run_cfg.symbols,
        year=year,
        month=month,
        timeframe=timeframe,
        steps_per_bar=steps_per_bar,
        seed=seed,
        root=run_cfg.storage_root,
        exchange_id=run_cfg.exchange_id,
    )
    # Event spacing ≈ bar / steps (1h / 60 → 60s).
    poll_sec = (3600.0 if timeframe == "1h" else 4 * 3600.0) / float(steps_per_bar)
    result, rows = run_realtime_paper(
        run_cfg,
        events,
        seed=seed,
        initial_equity=initial_equity,
        poll_interval_sec=poll_sec,
    )
    meta = {
        "mode": "hist_ohlc_synth_bbo_event_driven",
        "year": year,
        "month": month,
        "timeframe": timeframe,
        "steps_per_bar": steps_per_bar,
        "n_events": len(events),
        "fill_prob": run_cfg.sim.fill_prob,
        "adverse_frac": run_cfg.sim.adverse_frac,
        "days_in_month": calendar.monthrange(year, month)[1],
        "interpretation_allowed": False,
        "note": (
            "BBO synthesized from OHLC (not archived L2). "
            "Hold times are wall-clock on the event clock; "
            "PnL is not a live-edge claim."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    # Strengthen notes on result
    result.notes = (
        f"{result.notes} Hist-synth {year:04d}-{month:02d} {timeframe}→"
        f"~{steps_per_bar}/bar BBO; fill_prob={run_cfg.sim.fill_prob}, "
        f"adverse_frac={run_cfg.sim.adverse_frac}. Not archived L2."
    )
    return result, rows, meta


def save_month_events_artifacts(
    result: RealtimeResult,
    rows: list[dict],
    meta: dict,
    *,
    root: str | Path | None = None,
    exchange_id: str = "okx",
    basename: str,
    persist_rows: bool = False,
) -> dict[str, str]:
    """Persist summary+meta. Skip full event CSV by default (can be tens of MB)."""
    import json
    from dataclasses import asdict

    out = Path(root or "data") / exchange_id / "market_making"
    out.mkdir(parents=True, exist_ok=True)
    summary = out / f"{basename}.json"
    slim = {k: v for k, v in asdict(result).items() if k != "holds_sec"}
    slim["holds_sec_sample"] = list(result.holds_sec[:50])
    summary.write_text(json.dumps(slim, indent=2, default=str), encoding="utf-8")
    paths: dict[str, str] = {"summary": str(summary)}

    if persist_rows and rows:
        csv = out / f"{basename}.csv"
        pd.DataFrame(rows).to_csv(csv, index=False)
        paths["csv"] = str(csv)

    meta_path = out / f"{basename}_meta.json"
    payload = {
        **meta,
        "result": {k: v for k, v in result.__dict__.items() if k != "holds_sec"},
    }
    meta_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    paths["meta"] = str(meta_path)
    return paths
