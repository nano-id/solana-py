"""Johansen multivariate cointegration screens."""

from __future__ import annotations

import itertools
import logging
from typing import Iterable

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

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


def johansen_basket(
    log_prices: pd.DataFrame,
    symbols: list[str],
    *,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> dict:
    """Run Johansen trace test; return summary dict."""
    data = log_prices[symbols].to_numpy()
    res = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)
    # Trace stats vs 95% critical values (column index 1 in cvt)
    trace = np.asarray(res.lr1, dtype=float)
    cv95 = np.asarray(res.cvt[:, 1], dtype=float)
    eig = np.asarray(res.lr2, dtype=float)
    n_coint = int(np.sum(trace > cv95))
    return {
        "symbols": "|".join(symbols),
        "n_symbols": len(symbols),
        "n_obs": int(len(log_prices)),
        "det_order": det_order,
        "k_ar_diff": k_ar_diff,
        "n_coint_relations_95": n_coint,
        "has_coint_95": n_coint >= 1,
        "trace_stats": ";".join(f"{v:.4f}" for v in trace),
        "trace_cv95": ";".join(f"{v:.4f}" for v in cv95),
        "max_eig_stats": ";".join(f"{v:.4f}" for v in eig),
        "trace_r0": float(trace[0]) if len(trace) else float("nan"),
        "cv95_r0": float(cv95[0]) if len(cv95) else float("nan"),
    }


def scan_johansen_baskets(
    config: DataConfig | None = None,
    *,
    candidates: Iterable[str] | None = None,
    sizes: tuple[int, ...] = (3, 4),
    exchange_id: str | None = None,
    persist: bool = True,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> pd.DataFrame:
    """Test all combinations of `sizes` among candidate symbols."""
    config = config or load_config()
    eid = resolve_exchange_id(config, exchange_id)
    store = ParquetStore(config.storage.root, eid, config.ohlcv.timeframe)
    cands = list(candidates or DEFAULT_CANDIDATES)

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

    rows: list[dict] = []
    for k in sizes:
        if len(available) < k:
            continue
        for combo in itertools.combinations(available, k):
            syms = list(combo)
            try:
                aligned = _aligned_log_closes(
                    frames, syms, min_obs=config.cointegration.min_obs
                )
                summary = johansen_basket(
                    aligned, syms, det_order=det_order, k_ar_diff=k_ar_diff
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Johansen failed for %s: %s", syms, exc)
                continue
            rows.append(summary)
            logger.info(
                "Johansen %s: n_coint@95=%d (trace_r0=%.3f cv=%.3f)",
                summary["symbols"],
                summary["n_coint_relations_95"],
                summary["trace_r0"],
                summary["cv95_r0"],
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ["has_coint_95", "n_coint_relations_95", "trace_r0"],
            ascending=[False, False, False],
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
            },
            basename="johansen_baskets",
        )
    return df
