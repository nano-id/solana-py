"""Dual-null Engle-Granger p-values for rolling FDR survivor pairs.

Nulls (same spirit as Johansen dual-null):
  1. independent Gaussian random walks → p_rw
  2. block-bootstrap of joint Δlog return vectors → p_bb

EG is left-tailed in the test statistic (more negative ⇒ stronger rejection).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from pairs_trading.cointegration.engine import evaluate_pair
from pairs_trading.cointegration.hf_retest import load_rolling_fdr_hits
from pairs_trading.cointegration.johansen_mc import (
    block_bootstrap_levels,
    default_block_size,
    simulate_independent_random_walks,
)
from pairs_trading.cointegration.persist import save_scan_results
from pairs_trading.cointegration.scanner import load_symbol_frame, resolve_exchange_id
from pairs_trading.data.config import DataConfig, load_config
from pairs_trading.data.storage import ParquetStore

logger = logging.getLogger(__name__)


def _iso_to_ms(iso: str) -> int:
    dt = datetime.strptime(str(iso), "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _slice_ohlcv(df: pd.DataFrame, start_ms: int, end_ms: int) -> pd.DataFrame:
    out = df[(df["timestamp"] >= start_ms) & (df["timestamp"] < end_ms)].copy()
    return out.reset_index(drop=True)


def _eg_stat(y: np.ndarray, x: np.ndarray, *, autolag: str | None = "aic") -> float:
    score, _, _ = coint(y, x, autolag=autolag)
    return float(score)


def _left_tail_pvalue(observed: float, null_stats: np.ndarray, n_sims: int) -> dict:
    valid = null_stats[np.isfinite(null_stats)]
    n_valid = int(len(valid))
    if n_valid == 0 or not np.isfinite(observed):
        return {
            "pvalue": 1.0,
            "n_valid": 0,
            "null_mean": float("nan"),
            "null_p05": float("nan"),
            "null_p01": float("nan"),
        }
    # More negative than observed ⇒ at least as extreme
    n_le = int(np.sum(valid <= observed))
    return {
        "pvalue": float((1.0 + n_le) / (n_valid + 1.0)),
        "n_valid": n_valid,
        "null_mean": float(np.mean(valid)),
        "null_p05": float(np.quantile(valid, 0.05)),
        "null_p01": float(np.quantile(valid, 0.01)),
    }


def eg_indep_rw_null_stats(
    n_obs: int,
    *,
    n_sims: int,
    seed: int,
    autolag: str | None = "aic",
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty(n_sims, dtype=float)
    for i in range(n_sims):
        walks = simulate_independent_random_walks(n_obs, 2, rng)
        try:
            out[i] = _eg_stat(walks[:, 0], walks[:, 1], autolag=autolag)
        except Exception:  # noqa: BLE001
            out[i] = np.nan
    return out


def eg_block_bootstrap_null_stats(
    log_yx: np.ndarray,
    *,
    n_sims: int,
    seed: int,
    block_size: int | None = None,
    autolag: str | None = "aic",
) -> np.ndarray:
    """log_yx: (T, 2) aligned log prices [y, x]."""
    rng = np.random.default_rng(seed)
    out = np.empty(n_sims, dtype=float)
    for i in range(n_sims):
        try:
            paths = block_bootstrap_levels(log_yx, rng, block_size=block_size)
            out[i] = _eg_stat(paths[:, 0], paths[:, 1], autolag=autolag)
        except Exception:  # noqa: BLE001
            out[i] = np.nan
    return out


def dual_null_eg_pair(
    log_y: np.ndarray,
    log_x: np.ndarray,
    *,
    n_sims: int = 2000,
    seed: int = 0,
    block_size: int | None = None,
    autolag: str | None = "aic",
) -> dict:
    """Compare MacKinnon / RW / block-bootstrap p-values for one EG pair window."""
    y = np.asarray(log_y, dtype=float).ravel()
    x = np.asarray(log_x, dtype=float).ravel()
    if len(y) != len(x) or len(y) < 30:
        raise ValueError("Need aligned log series with length >= 30")
    log_yx = np.column_stack([y, x])
    obs_stat, table_p, _ = coint(y, x, autolag=autolag)
    obs_stat = float(obs_stat)
    table_p = float(table_p)

    bs = block_size if block_size is not None else default_block_size(len(y) - 1)
    rw_stats = eg_indep_rw_null_stats(len(y), n_sims=n_sims, seed=seed, autolag=autolag)
    bb_stats = eg_block_bootstrap_null_stats(
        log_yx,
        n_sims=n_sims,
        seed=seed + 10_000,
        block_size=bs,
        autolag=autolag,
    )
    rw = _left_tail_pvalue(obs_stat, rw_stats, n_sims)
    bb = _left_tail_pvalue(obs_stat, bb_stats, n_sims)
    p_rw = float(rw["pvalue"])
    p_bb = float(bb["pvalue"])
    return {
        "test_statistic": obs_stat,
        "pvalue_table": table_p,
        "pvalue_mc": p_rw,
        "pvalue_bb": p_bb,
        "p_ratio_bb_over_rw": (p_bb / p_rw) if p_rw > 0 else float("inf"),
        "rw_null_p05": float(rw["null_p05"]),
        "bb_null_p05": float(bb["null_p05"]),
        "block_size": int(bs),
        "n_sims": n_sims,
        "n_obs": int(len(y)),
    }


def dual_null_rolling_fdr_pairs(
    config: DataConfig | None = None,
    *,
    exchange_id: str | None = None,
    window_days: tuple[int, ...] = (180, 365),
    n_sims: int = 2000,
    seed: int = 42,
    block_size: int | None = None,
    persist: bool = True,
    autolag: str | None = "aic",
) -> pd.DataFrame:
    """Apply dual-null EG to every daily rolling FDR survivor row."""
    config = config or load_config()
    eid = resolve_exchange_id(config, exchange_id)
    store = ParquetStore(config.storage.root, eid, config.ohlcv.timeframe)
    hits = load_rolling_fdr_hits(
        config,
        exchange_id=eid,
        window_days=window_days,
        timeframe=config.ohlcv.timeframe,
    )
    if hits.empty:
        return pd.DataFrame()

    symbols = sorted(
        {*(hits["y_symbol"].astype(str)), *(hits["x_symbol"].astype(str))}
    )
    frames: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            frames[sym] = load_symbol_frame(store, sym)
        except FileNotFoundError:
            logger.warning("Missing OHLCV for %s — skip pairs using it", sym)

    rows: list[dict] = []
    for i, hit in hits.iterrows():
        y_sym, x_sym = str(hit["y_symbol"]), str(hit["x_symbol"])
        if y_sym not in frames or x_sym not in frames:
            continue
        w0, w1 = _iso_to_ms(hit["window_start"]), _iso_to_ms(hit["window_end"])
        y_df = _slice_ohlcv(frames[y_sym], w0, w1)
        x_df = _slice_ohlcv(frames[x_sym], w0, w1)
        min_obs = max(60, int(hit.get("window_days", 180)) - 5)
        try:
            eg = evaluate_pair(
                y_sym,
                x_sym,
                y_df,
                x_df,
                min_obs=min_obs,
                price_space=config.cointegration.price_space,
                autolag=autolag,
            )
        except ValueError as exc:
            logger.warning("EG failed %s|%s: %s", y_sym, x_sym, exc)
            continue

        # Aligned log prices for bootstrap (same join as evaluate_pair)
        from pairs_trading.cointegration.engine import align_close_prices

        y_log, x_log = align_close_prices(
            y_df, x_df, min_obs=min_obs, price_space=config.cointegration.price_space
        )
        pair_seed = seed + 31 * (1 + int(i) if isinstance(i, (int, np.integer)) else len(rows))
        logger.info(
            "EG dual-null %s|%s %s→%s (n=%d sims=%d) …",
            y_sym,
            x_sym,
            hit["window_start"],
            hit["window_end"],
            len(y_log),
            n_sims,
        )
        dual = dual_null_eg_pair(
            y_log.to_numpy(),
            x_log.to_numpy(),
            n_sims=n_sims,
            seed=pair_seed,
            block_size=block_size,
            autolag=autolag,
        )
        contains_bnb = y_sym.upper().startswith("BNB/") or x_sym.upper().startswith("BNB/")
        rows.append(
            {
                "window_start": hit["window_start"],
                "window_end": hit["window_end"],
                "window_days": int(hit.get("window_days", 0)),
                "y_symbol": y_sym,
                "x_symbol": x_sym,
                "pair": f"{y_sym}|{x_sym}",
                "contains_bnb": contains_bnb,
                "daily_pvalue": float(hit["pvalue"]),
                "daily_pvalue_fdr": float(hit.get("pvalue_fdr", float("nan"))),
                "daily_half_life": float(hit.get("half_life", float("nan"))),
                "hedge_ratio": eg.hedge_ratio,
                **dual,
            }
        )
        logger.info(
            "  table=%.4f rw=%.4f bb=%.4f ratio=%.2f  (stat=%.3f)",
            dual["pvalue_table"],
            dual["pvalue_mc"],
            dual["pvalue_bb"],
            dual["p_ratio_bb_over_rw"],
            dual["test_statistic"],
        )

    out = pd.DataFrame(rows)
    if persist and not out.empty:
        save_scan_results(
            out,
            root=config.storage.root,
            exchange_id=eid,
            timeframe=config.ohlcv.timeframe,
            mode=config.cointegration.persist_mode,
            meta={
                "kind": "eg_dual_null_rolling_fdr",
                "n_sims": n_sims,
                "seed": seed,
                "block_size": block_size or "sqrt_T",
                "n_pairs": len(out),
                "n_bnb": int(out["contains_bnb"].sum()),
                "n_bnb_free": int((~out["contains_bnb"]).sum()),
                "median_ratio_bnb": float(
                    out.loc[out["contains_bnb"], "p_ratio_bb_over_rw"].median()
                )
                if out["contains_bnb"].any()
                else None,
                "median_ratio_bnb_free": float(
                    out.loc[~out["contains_bnb"], "p_ratio_bb_over_rw"].median()
                )
                if (~out["contains_bnb"]).any()
                else None,
            },
            basename="eg_dual_null_rolling_fdr",
        )
    return out


def summarize_eg_dual_by_bnb(df: pd.DataFrame) -> pd.DataFrame:
    """Group dual-null ratios by BNB involvement."""
    if df.empty:
        return df
    rows = []
    for label, mask in (
        ("bnb", df["contains_bnb"]),
        ("bnb_free", ~df["contains_bnb"]),
        ("all", pd.Series(True, index=df.index)),
    ):
        g = df.loc[mask]
        if g.empty:
            continue
        rows.append(
            {
                "group": label,
                "n": len(g),
                "median_p_rw": float(g["pvalue_mc"].median()),
                "median_p_bb": float(g["pvalue_bb"].median()),
                "median_ratio": float(g["p_ratio_bb_over_rw"].median()),
                "min_ratio": float(g["p_ratio_bb_over_rw"].min()),
                "max_ratio": float(g["p_ratio_bb_over_rw"].max()),
                "n_bb_lt_0_05": int((g["pvalue_bb"] < 0.05).sum()),
                "n_rw_lt_0_05": int((g["pvalue_mc"] < 0.05).sum()),
            }
        )
    return pd.DataFrame(rows)
