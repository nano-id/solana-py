"""Higher-frequency retest of daily rolling FDR survivor pairs."""

from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd

from pairs_trading.cointegration.engine import evaluate_pair
from pairs_trading.cointegration.persist import save_scan_results
from pairs_trading.cointegration.scanner import load_symbol_frame, resolve_exchange_id
from pairs_trading.data.config import DataConfig, load_config
from pairs_trading.data.storage import ParquetStore

logger = logging.getLogger(__name__)

# Hours per bar for common crypto timeframes.
_TF_HOURS = {
    "1h": 1.0,
    "4h": 4.0,
    "1d": 24.0,
    "1D": 24.0,
}


def timeframe_hours(timeframe: str) -> float:
    if timeframe in _TF_HOURS:
        return _TF_HOURS[timeframe]
    raise ValueError(f"Unsupported timeframe for day conversion: {timeframe}")


def _iso_to_ms(iso: str) -> int:
    dt = datetime.strptime(iso, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _slice_ohlcv(df: pd.DataFrame, start_ms: int, end_ms: int) -> pd.DataFrame:
    out = df[(df["timestamp"] >= start_ms) & (df["timestamp"] < end_ms)].copy()
    return out.reset_index(drop=True)


def retest_daily_fdr_pairs_at_hf(
    daily_hits: pd.DataFrame,
    config: DataConfig | None = None,
    *,
    hf_timeframe: str = "4h",
    exchange_id: str | None = None,
    persist: bool = True,
    min_obs: int | None = None,
) -> pd.DataFrame:
    """Re-run EG on calendar windows of daily FDR hits using higher-freq bars.

    Compares half-life in *days* (daily bars vs hf_bars * hours/24) so short
    mean-reversion that daily bars smear can be diagnosed.
    """
    config = config or load_config()
    eid = resolve_exchange_id(config, exchange_id)
    store = ParquetStore(config.storage.root, eid, hf_timeframe)
    hours = timeframe_hours(hf_timeframe)
    bars_per_day = 24.0 / hours

    needed = daily_hits.loc[daily_hits["significant_fdr"]].copy()
    if needed.empty:
        return pd.DataFrame()

    symbols: set[str] = set()
    for _, row in needed.iterrows():
        symbols.add(str(row["y_symbol"]))
        symbols.add(str(row["x_symbol"]))

    frames: dict[str, pd.DataFrame] = {}
    for sym in sorted(symbols):
        try:
            frames[sym] = load_symbol_frame(store, sym)
        except FileNotFoundError:
            logger.warning("HF retest skip %s — no %s OHLCV", sym, hf_timeframe)

    rows: list[dict] = []
    for _, hit in needed.iterrows():
        y_sym, x_sym = str(hit["y_symbol"]), str(hit["x_symbol"])
        if y_sym not in frames or x_sym not in frames:
            continue
        w_start = _iso_to_ms(str(hit["window_start"]))
        w_end = _iso_to_ms(str(hit["window_end"]))
        y_df = _slice_ohlcv(frames[y_sym], w_start, w_end)
        x_df = _slice_ohlcv(frames[x_sym], w_start, w_end)
        # ~window_days * bars_per_day, with a floor
        window_days = int(hit.get("window_days", (w_end - w_start) / 86_400_000))
        floor = min_obs if min_obs is not None else max(120, int(0.7 * window_days * bars_per_day))
        try:
            eg = evaluate_pair(
                y_sym,
                x_sym,
                y_df,
                x_df,
                min_obs=floor,
                price_space=config.cointegration.price_space,
            )
        except ValueError as exc:
            logger.warning(
                "HF retest failed %s|%s %s→%s: %s",
                y_sym,
                x_sym,
                hit["window_start"],
                hit["window_end"],
                exc,
            )
            continue

        daily_hl = float(hit["half_life"]) if pd.notna(hit.get("half_life")) else float("nan")
        hf_hl_bars = float(eg.half_life)
        hf_hl_days = hf_hl_bars * (hours / 24.0) if hf_hl_bars < float("inf") else float("inf")
        row = asdict(eg)
        row.update(
            {
                "pair": f"{y_sym}|{x_sym}",
                "window_start": hit["window_start"],
                "window_end": hit["window_end"],
                "window_days": window_days,
                "hf_timeframe": hf_timeframe,
                "daily_pvalue": float(hit["pvalue"]),
                "daily_pvalue_fdr": float(hit.get("pvalue_fdr", float("nan"))),
                "daily_half_life_days": daily_hl,
                "hf_half_life_bars": hf_hl_bars,
                "hf_half_life_days": hf_hl_days,
                "hl_ratio_hf_over_daily": (
                    hf_hl_days / daily_hl
                    if daily_hl and daily_hl < float("inf") and hf_hl_days < float("inf")
                    else float("nan")
                ),
            }
        )
        rows.append(row)
        logger.info(
            "HF %s %s|%s %s→%s: p=%.4g HL_d=%.2f (daily HL_d=%.2f, ratio=%.2f)",
            hf_timeframe,
            y_sym,
            x_sym,
            hit["window_start"],
            hit["window_end"],
            eg.pvalue,
            hf_hl_days,
            daily_hl,
            row["hl_ratio_hf_over_daily"],
        )

    out = pd.DataFrame(rows)
    if persist and not out.empty:
        save_scan_results(
            out,
            root=config.storage.root,
            exchange_id=eid,
            timeframe=hf_timeframe,
            mode=config.cointegration.persist_mode,
            meta={
                "kind": "hf_retest_daily_fdr",
                "hf_timeframe": hf_timeframe,
                "n_daily_hits": int(len(needed)),
                "n_retested": int(len(out)),
                "n_raw_sig": int((out["pvalue"] < config.cointegration.pvalue_threshold).sum()),
            },
            basename=f"hf_retest_{hf_timeframe}_from_1d_fdr",
        )
    return out


def load_rolling_fdr_hits(
    config: DataConfig,
    *,
    exchange_id: str | None = None,
    window_days: Iterable[int] = (180, 365),
    timeframe: str = "1d",
) -> pd.DataFrame:
    """Load latest rolling_* FDR survivors from disk and stack them."""
    eid = resolve_exchange_id(config, exchange_id)
    root = config.storage.root / eid / "cointegration"
    frames: list[pd.DataFrame] = []
    for wd in window_days:
        path = root / f"rolling_{wd}d_none_latest_{timeframe}.parquet"
        if not path.is_file():
            logger.warning("Missing rolling artifact: %s", path)
            continue
        df = pd.read_parquet(path)
        if "significant_fdr" not in df.columns:
            continue
        hit = df[df["significant_fdr"]].copy()
        if "window_days" not in hit.columns:
            hit["window_days"] = wd
        frames.append(hit)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
