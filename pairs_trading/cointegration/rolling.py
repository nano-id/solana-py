"""Rolling-window Engle-Granger screens for regime discovery."""

from __future__ import annotations

import itertools
import logging
from dataclasses import asdict
from datetime import datetime, timezone

import pandas as pd

from pairs_trading.cointegration.engine import evaluate_pair
from pairs_trading.cointegration.fdr import apply_benjamini_hochberg
from pairs_trading.cointegration.neutralize import build_factor_residual_frames
from pairs_trading.cointegration.persist import save_scan_results
from pairs_trading.cointegration.scanner import (
    load_symbol_frame,
    resolve_exchange_id,
    resolve_universe_symbols,
)
from pairs_trading.data.config import DataConfig, load_config
from pairs_trading.data.storage import ParquetStore

logger = logging.getLogger(__name__)


def _slice_ohlcv(df: pd.DataFrame, start_ms: int, end_ms: int) -> pd.DataFrame:
    out = df[(df["timestamp"] >= start_ms) & (df["timestamp"] < end_ms)].copy()
    return out.reset_index(drop=True)


def iter_windows(
    start_ms: int,
    end_ms: int,
    *,
    window_days: int,
    step_days: int,
) -> list[tuple[int, int, str, str]]:
    """Return [(start_ms, end_ms, start_iso, end_iso), ...] non-empty windows."""
    window_ms = window_days * 86_400_000
    step_ms = step_days * 86_400_000
    out: list[tuple[int, int, str, str]] = []
    cur = start_ms
    while cur + window_ms <= end_ms:
        w_end = cur + window_ms
        s_iso = datetime.fromtimestamp(cur / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        e_iso = datetime.fromtimestamp(w_end / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        out.append((cur, w_end, s_iso, e_iso))
        cur += step_ms
    return out


def common_time_bounds(frames: dict[str, pd.DataFrame]) -> tuple[int, int]:
    starts = [int(df["timestamp"].min()) for df in frames.values() if not df.empty]
    ends = [int(df["timestamp"].max()) for df in frames.values() if not df.empty]
    if not starts or not ends:
        raise ValueError("No OHLCV frames for rolling bounds")
    return max(starts), min(ends) + 86_400_000  # exclusive end ~ last bar + 1d


def scan_rolling_windows(
    config: DataConfig | None = None,
    *,
    window_days: int = 180,
    step_days: int = 90,
    exchange_id: str | None = None,
    symbols: list[str] | None = None,
    preprocess: str | None = None,
    persist: bool = True,
    min_obs: int | None = None,
) -> pd.DataFrame:
    """Run pairwise EG+FDR inside each rolling window; return stacked results."""
    config = config or load_config()
    eid = resolve_exchange_id(config, exchange_id)
    store = ParquetStore(config.storage.root, eid, config.ohlcv.timeframe)
    alpha = config.cointegration.pvalue_threshold
    preprocess = preprocess or config.cointegration.preprocess
    factor_symbol = config.cointegration.factor_symbol
    # Short windows need a looser bar floor than full-sample scans.
    min_obs = min_obs if min_obs is not None else min(
        config.cointegration.min_obs, max(60, window_days - 5)
    )

    universe, universe_reason = resolve_universe_symbols(config, store, symbols=symbols)
    load_syms = list(universe)
    if preprocess == "btc_residual" and factor_symbol not in load_syms:
        load_syms.append(factor_symbol)

    raw_frames: dict[str, pd.DataFrame] = {}
    for sym in load_syms:
        try:
            raw_frames[sym] = load_symbol_frame(store, sym)
        except FileNotFoundError:
            logger.warning("Rolling skip %s — no OHLCV", sym)

    if len(raw_frames) < 2:
        return pd.DataFrame()

    t0, t1 = common_time_bounds(raw_frames)
    windows = iter_windows(t0, t1, window_days=window_days, step_days=step_days)
    logger.info(
        "Rolling scan: %d windows of %dd step %dd | preprocess=%s | universe=%s",
        len(windows),
        window_days,
        step_days,
        preprocess,
        universe_reason,
    )

    all_rows: list[dict] = []
    for w_start, w_end, s_iso, e_iso in windows:
        sliced = {
            sym: _slice_ohlcv(df, w_start, w_end)
            for sym, df in raw_frames.items()
            if len(_slice_ohlcv(df, w_start, w_end)) >= min_obs
        }
        # Re-check min_obs after slice
        sliced = {
            sym: df
            for sym, df in sliced.items()
            if len(df) >= min_obs
        }
        if preprocess == "btc_residual":
            if factor_symbol not in sliced:
                logger.info("Window %s→%s skip — factor missing/short", s_iso, e_iso)
                continue
            try:
                frames, _ = build_factor_residual_frames(
                    sliced,
                    factor_symbol=factor_symbol,
                    min_obs=min_obs,
                )
            except FileNotFoundError:
                continue
            eval_price_space = "level"
        elif preprocess == "none":
            frames = sliced
            eval_price_space = config.cointegration.price_space
        else:
            raise ValueError(f"Unknown preprocess={preprocess!r}")

        if len(frames) < 2:
            continue

        rows: list[dict] = []
        for y_sym, x_sym in itertools.combinations(sorted(frames.keys()), 2):
            try:
                hit = evaluate_pair(
                    y_sym,
                    x_sym,
                    frames[y_sym],
                    frames[x_sym],
                    min_obs=min_obs,
                    price_space=eval_price_space,
                )
            except ValueError:
                continue
            row = asdict(hit)
            row["pair"] = f"{hit.y_symbol}|{hit.x_symbol}"
            row["window_start"] = s_iso
            row["window_end"] = e_iso
            row["window_days"] = window_days
            row["preprocess"] = preprocess
            row["price_space"] = (
                "btc_residual" if preprocess == "btc_residual" else hit.price_space
            )
            rows.append(row)

        if not rows:
            continue
        wdf = pd.DataFrame(rows)
        wdf = apply_benjamini_hochberg(wdf, pvalue_col="pvalue", alpha=alpha)
        n_fdr = int(wdf["significant_fdr"].sum())
        logger.info(
            "Window %s→%s: pairs=%d raw=%d fdr=%d",
            s_iso,
            e_iso,
            len(wdf),
            int(wdf["significant_raw"].sum()),
            n_fdr,
        )
        all_rows.extend(wdf.to_dict(orient="records"))

    out = pd.DataFrame(all_rows)
    if not out.empty:
        out = out.sort_values(
            ["window_start", "pvalue_fdr", "pvalue"], kind="mergesort"
        ).reset_index(drop=True)

    if persist and not out.empty:
        save_scan_results(
            out,
            root=config.storage.root,
            exchange_id=eid,
            timeframe=config.ohlcv.timeframe,
            mode=config.cointegration.persist_mode,
            meta={
                "kind": "rolling",
                "window_days": window_days,
                "step_days": step_days,
                "preprocess": preprocess,
                "n_windows": len(windows),
                "n_rows": len(out),
                "n_fdr_hits": int(out["significant_fdr"].sum()),
                "universe_reason": universe_reason,
            },
            basename=f"rolling_{window_days}d_{preprocess}",
        )
    return out


def summarize_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """Per-window counts of raw/FDR hits."""
    if df.empty:
        return df
    g = (
        df.groupby(["window_start", "window_end", "window_days"], as_index=False)
        .agg(
            n_pairs=("pair", "count"),
            n_raw=("significant_raw", "sum"),
            n_fdr=("significant_fdr", "sum"),
        )
        .sort_values("window_start")
    )
    return g
