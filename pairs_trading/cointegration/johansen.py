"""Johansen multivariate cointegration screens."""

from __future__ import annotations

import itertools
import logging
import random
from typing import Iterable

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from pairs_trading.cointegration.fdr import apply_bh_and_by
from pairs_trading.cointegration.persist import save_scan_results
from pairs_trading.cointegration.scanner import (
    load_symbol_frame,
    resolve_exchange_id,
)
from pairs_trading.data.config import DataConfig, load_config
from pairs_trading.data.storage import ParquetStore

logger = logging.getLogger(__name__)

DEFAULT_CANDIDATES = (
    "BNB/USDT",
    "OKB/USDT",
    "SOL/USDT",
    "UNI/USDT",
    "ARB/USDT",
    "WLD/USDT",
)


def _aligned_log_closes(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    *,
    min_obs: int,
) -> pd.DataFrame:
    """Inner-join closes on timestamp, return log-price columns named by symbol."""
    merged: pd.DataFrame | None = None
    for sym in symbols:
        if sym not in frames:
            raise FileNotFoundError(f"Missing frame for {sym}")
        piece = frames[sym][["timestamp", "close"]].rename(columns={"close": sym})
        piece[sym] = np.log(piece[sym].astype(float))
        merged = piece if merged is None else merged.merge(piece, on="timestamp", how="inner")
    assert merged is not None
    merged = merged.dropna().sort_values("timestamp")
    if len(merged) < min_obs:
        raise ValueError(f"Insufficient overlap for {symbols}: {len(merged)} < {min_obs}")
    return merged


def approx_trace_pvalue(trace: float, cv90: float, cv95: float, cv99: float) -> float:
    """Approximate p-value for Johansen trace (H0: r=0) from critical values.

    statsmodels.coint_johansen does not return MacKinnon p-values; we interpolate
    against the 90/95/99% CV table so BH-FDR can be applied consistently with EG.
    This is deliberately coarse/conservative — good enough for multiple-testing triage.
    """
    if not np.isfinite(trace):
        return 1.0
    if trace >= cv99:
        return 0.005
    if trace >= cv95:
        w = (trace - cv95) / max(cv99 - cv95, 1e-12)
        return float(0.05 - 0.045 * w)  # 0.05 → 0.005
    if trace >= cv90:
        w = (trace - cv90) / max(cv95 - cv90, 1e-12)
        return float(0.10 - 0.05 * w)  # 0.10 → 0.05
    if cv90 <= 0:
        return 1.0
    w = max(0.0, min(1.0, trace / cv90))
    return float(1.0 - 0.90 * w)  # 0 → 1.0, cv90 → 0.10


def johansen_basket(
    log_prices: pd.DataFrame,
    symbols: list[str],
    *,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> dict:
    """Run Johansen trace test; return summary dict incl. approx p-value for r=0."""
    data = log_prices[symbols].to_numpy()
    res = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)
    trace = np.asarray(res.lr1, dtype=float)
    cvt = np.asarray(res.cvt, dtype=float)  # columns: 90%, 95%, 99%
    eig = np.asarray(res.lr2, dtype=float)
    cv90, cv95, cv99 = float(cvt[0, 0]), float(cvt[0, 1]), float(cvt[0, 2])
    trace_r0 = float(trace[0]) if len(trace) else float("nan")
    p_r0 = approx_trace_pvalue(trace_r0, cv90, cv95, cv99)
    n_coint = int(np.sum(trace > cvt[:, 1]))
    return {
        "symbols": "|".join(symbols),
        "n_symbols": len(symbols),
        "n_obs": int(len(log_prices)),
        "det_order": det_order,
        "k_ar_diff": k_ar_diff,
        "n_coint_relations_95": n_coint,
        "has_coint_95": n_coint >= 1,
        "pvalue": p_r0,  # H0: r=0 approx — used for BH-FDR
        "trace_stats": ";".join(f"{v:.4f}" for v in trace),
        "trace_cv95": ";".join(f"{v:.4f}" for v in cvt[:, 1]),
        "max_eig_stats": ";".join(f"{v:.4f}" for v in eig),
        "trace_r0": trace_r0,
        "cv90_r0": cv90,
        "cv95_r0": cv95,
        "cv99_r0": cv99,
        "contains_bnb": any(s.upper().startswith("BNB/") for s in symbols),
    }


def _run_baskets(
    frames: dict[str, pd.DataFrame],
    baskets: list[tuple[str, ...]],
    *,
    min_obs: int,
    det_order: int,
    k_ar_diff: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for combo in baskets:
        syms = list(combo)
        try:
            aligned = _aligned_log_closes(frames, syms, min_obs=min_obs)
            summary = johansen_basket(
                aligned, syms, det_order=det_order, k_ar_diff=k_ar_diff
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Johansen failed for %s: %s", syms, exc)
            continue
        rows.append(summary)
        logger.info(
            "Johansen %s: n_coint@95=%d p≈%.4f (trace_r0=%.3f cv95=%.3f)",
            summary["symbols"],
            summary["n_coint_relations_95"],
            summary["pvalue"],
            summary["trace_r0"],
            summary["cv95_r0"],
        )
    return pd.DataFrame(rows)


def scan_johansen_baskets(
    config: DataConfig | None = None,
    *,
    candidates: Iterable[str] | None = None,
    sizes: tuple[int, ...] = (3, 4),
    exchange_id: str | None = None,
    persist: bool = True,
    det_order: int = 0,
    k_ar_diff: int = 1,
    alpha: float | None = None,
) -> pd.DataFrame:
    """Test all combinations of `sizes` among candidate symbols + BH-FDR."""
    config = config or load_config()
    eid = resolve_exchange_id(config, exchange_id)
    store = ParquetStore(config.storage.root, eid, config.ohlcv.timeframe)
    cands = list(candidates or DEFAULT_CANDIDATES)
    alpha = config.cointegration.pvalue_threshold if alpha is None else alpha

    frames: dict[str, pd.DataFrame] = {}
    for sym in cands:
        try:
            frames[sym] = load_symbol_frame(store, sym)
        except FileNotFoundError:
            logger.warning("Johansen skip %s — no OHLCV", sym)
    available = [s for s in cands if s in frames]
    if len(available) < min(sizes):
        raise FileNotFoundError(
            f"Need at least {min(sizes)} symbols with data; have {available}"
        )

    baskets: list[tuple[str, ...]] = []
    for k in sizes:
        if len(available) >= k:
            baskets.extend(itertools.combinations(available, k))

    df = _run_baskets(
        frames,
        baskets,
        min_obs=config.cointegration.min_obs,
        det_order=det_order,
        k_ar_diff=k_ar_diff,
    )
    if not df.empty:
        df = apply_bh_and_by(df, pvalue_col="pvalue", alpha=alpha)
        df = df.sort_values(
            ["significant_fdr", "pvalue_fdr", "trace_r0"],
            ascending=[False, True, False],
        ).reset_index(drop=True)

    if persist and not df.empty:
        save_scan_results(
            df,
            root=config.storage.root,
            exchange_id=eid,
            timeframe=config.ohlcv.timeframe,
            mode=config.cointegration.persist_mode,
            meta={
                "kind": "johansen",
                "candidates": available,
                "sizes": list(sizes),
                "det_order": det_order,
                "k_ar_diff": k_ar_diff,
                "n_baskets": len(df),
                "n_with_coint_95": int(df["has_coint_95"].sum()),
                "n_significant_fdr": int(df["significant_fdr"].sum()),
                "n_significant_by": int(df["significant_by"].sum()),
                "fdr_methods": ["benjamini_hochberg", "benjamini_yekutieli"],
            },
            basename="johansen_baskets",
        )
    return df


def scan_johansen_random_bnb_free(
    config: DataConfig | None = None,
    *,
    pool: Iterable[str] | None = None,
    n_baskets: int = 20,
    sizes: tuple[int, ...] = (3, 4),
    seed: int = 42,
    exchange_id: str | None = None,
    persist: bool = True,
    exclude_substr: str = "BNB/",
) -> pd.DataFrame:
    """Random 3/4-tuples from a BNB-free pool — baseline cointegration rate."""
    config = config or load_config()
    eid = resolve_exchange_id(config, exchange_id)
    store = ParquetStore(config.storage.root, eid, config.ohlcv.timeframe)
    alpha = config.cointegration.pvalue_threshold

    if pool is None:
        # list_symbols returns stems like BTC_USDT → BTC/USDT
        pool = []
        for stem in store.list_symbols():
            parts = stem.split("_", 1)
            if len(parts) == 2:
                pool.append(f"{parts[0]}/{parts[1]}")

    pool = [s for s in pool if exclude_substr.upper() not in s.upper()]
    frames: dict[str, pd.DataFrame] = {}
    for sym in pool:
        try:
            frames[sym] = load_symbol_frame(store, sym)
        except FileNotFoundError:
            continue
    available = sorted(frames.keys())
    if len(available) < min(sizes):
        raise FileNotFoundError(f"BNB-free pool too small: {available}")

    rng = random.Random(seed)
    candidates: list[tuple[str, ...]] = []
    for k in sizes:
        candidates.extend(itertools.combinations(available, k))
    rng.shuffle(candidates)
    baskets = candidates[:n_baskets]

    df = _run_baskets(
        frames,
        baskets,
        min_obs=config.cointegration.min_obs,
        det_order=0,
        k_ar_diff=1,
    )
    if not df.empty:
        df = apply_bh_and_by(df, pvalue_col="pvalue", alpha=alpha)
        df = df.sort_values(["pvalue_fdr", "pvalue"]).reset_index(drop=True)

    if persist and not df.empty:
        save_scan_results(
            df,
            root=config.storage.root,
            exchange_id=eid,
            timeframe=config.ohlcv.timeframe,
            mode=config.cointegration.persist_mode,
            meta={
                "kind": "johansen_bnb_free_random",
                "pool": available,
                "n_requested": n_baskets,
                "n_baskets": len(df),
                "seed": seed,
                "n_with_coint_95": int(df["has_coint_95"].sum()),
                "n_significant_fdr": int(df["significant_fdr"].sum()),
                "n_significant_by": int(df["significant_by"].sum()),
                "hit_rate_raw_95": float(df["has_coint_95"].mean()),
                "hit_rate_fdr": float(df["significant_fdr"].mean()),
                "hit_rate_by": float(df["significant_by"].mean()),
                "fdr_methods": ["benjamini_hochberg", "benjamini_yekutieli"],
            },
            basename="johansen_bnb_free_random",
        )
    return df
