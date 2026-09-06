"""Monte Carlo null p-values for Johansen trace (H0: r=0).

Two nulls:
  1. independent Gaussian random walks (classic asymptotic table analogue)
  2. block-bootstrap of real return vectors (keeps cross-asset correlation /
     vol structure, destroys time-order / cointegrating relations)
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from pairs_trading.cointegration.johansen import (
    _aligned_log_closes,
    approx_trace_pvalue,
)
from pairs_trading.cointegration.persist import save_scan_results
from pairs_trading.cointegration.scanner import load_symbol_frame, resolve_exchange_id
from pairs_trading.data.config import DataConfig, load_config
from pairs_trading.data.storage import ParquetStore

logger = logging.getLogger(__name__)

NullKind = Literal["indep_rw", "block_bootstrap"]


def _trace_r0(
    data: np.ndarray,
    *,
    det_order: int,
    k_ar_diff: int,
) -> float:
    res = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)
    return float(np.asarray(res.lr1, dtype=float)[0])


def _pvalue_from_null(observed: float, null: np.ndarray, n_sims: int) -> dict[str, float | int]:
    valid = null[np.isfinite(null)]
    n_valid = int(len(valid))
    if n_valid == 0 or not np.isfinite(observed):
        return {
            "pvalue": 1.0,
            "n_sims": n_sims,
            "n_valid": n_valid,
            "null_mean": float("nan"),
            "null_p95": float("nan"),
            "null_p99": float("nan"),
        }
    n_ge = int(np.sum(valid >= observed))
    return {
        "pvalue": float((1.0 + n_ge) / (n_valid + 1.0)),
        "n_sims": n_sims,
        "n_valid": n_valid,
        "null_mean": float(np.mean(valid)),
        "null_p95": float(np.quantile(valid, 0.95)),
        "null_p99": float(np.quantile(valid, 0.99)),
    }


def simulate_independent_random_walks(
    n_obs: int,
    n_series: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """I(1) series with no cointegration: independent Gaussian random walks."""
    shocks = rng.normal(size=(n_obs, n_series))
    return np.cumsum(shocks, axis=0)


def default_block_size(n_returns: int) -> int:
    """~sqrt(T) blocks; floor 5 so short-run serial corr is not fully destroyed."""
    return int(max(5, min(n_returns, round(np.sqrt(n_returns)))))


def block_bootstrap_levels(
    log_prices: np.ndarray,
    rng: np.random.Generator,
    *,
    block_size: int | None = None,
) -> np.ndarray:
    """Rebuild I(1) paths by block-bootstrapping contemporaneous return vectors.

    Returns (Δlog P_t) keep the cross-sectional correlation / relative vols of
    the real basket; reshuffling blocks breaks the shared stochastic trend that
    would produce cointegration. Levels are cumsum from 0 (Johansen is
    invariant to additive level shifts).
    """
    if log_prices.ndim != 2 or log_prices.shape[0] < 3:
        raise ValueError("log_prices must be (T, k) with T>=3")
    rets = np.diff(log_prices.astype(float), axis=0)
    t_ret, k = rets.shape
    bs = block_size if block_size is not None else default_block_size(t_ret)
    bs = max(1, min(bs, t_ret))

    n_blocks = int(np.ceil(t_ret / bs))
    max_start = t_ret - bs
    pieces: list[np.ndarray] = []
    for _ in range(n_blocks):
        start = int(rng.integers(0, max_start + 1))
        pieces.append(rets[start : start + bs])
    boot = np.vstack(pieces)[:t_ret]
    # (T_ret+1, k) levels matching original n_obs
    return np.vstack([np.zeros((1, k)), np.cumsum(boot, axis=0)])


def monte_carlo_trace_null(
    *,
    n_obs: int,
    n_series: int,
    det_order: int = 0,
    k_ar_diff: int = 1,
    n_sims: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    """Independent-RW null distribution of Johansen trace under r=0."""
    rng = np.random.default_rng(seed)
    out = np.empty(n_sims, dtype=float)
    for i in range(n_sims):
        walks = simulate_independent_random_walks(n_obs, n_series, rng)
        try:
            out[i] = _trace_r0(walks, det_order=det_order, k_ar_diff=k_ar_diff)
        except Exception:  # noqa: BLE001
            out[i] = np.nan
    return out


def block_bootstrap_trace_null(
    log_prices: np.ndarray,
    *,
    det_order: int = 0,
    k_ar_diff: int = 1,
    n_sims: int = 2000,
    seed: int = 0,
    block_size: int | None = None,
) -> np.ndarray:
    """Correlation-preserving block-bootstrap null for Johansen trace."""
    rng = np.random.default_rng(seed)
    out = np.empty(n_sims, dtype=float)
    for i in range(n_sims):
        try:
            paths = block_bootstrap_levels(log_prices, rng, block_size=block_size)
            out[i] = _trace_r0(paths, det_order=det_order, k_ar_diff=k_ar_diff)
        except Exception:  # noqa: BLE001
            out[i] = np.nan
    return out


def monte_carlo_trace_pvalue(
    observed_trace: float,
    *,
    n_obs: int,
    n_series: int,
    det_order: int = 0,
    k_ar_diff: int = 1,
    n_sims: int = 2000,
    seed: int = 0,
) -> dict[str, float | int]:
    """One-sided MC p-value under independent RW null."""
    null = monte_carlo_trace_null(
        n_obs=n_obs,
        n_series=n_series,
        det_order=det_order,
        k_ar_diff=k_ar_diff,
        n_sims=n_sims,
        seed=seed,
    )
    raw = _pvalue_from_null(observed_trace, null, n_sims)
    return {
        "pvalue_mc": raw["pvalue"],
        "n_sims": raw["n_sims"],
        "n_valid": raw["n_valid"],
        "null_mean": raw["null_mean"],
        "null_p95": raw["null_p95"],
        "null_p99": raw["null_p99"],
    }


def block_bootstrap_trace_pvalue(
    observed_trace: float,
    log_prices: np.ndarray,
    *,
    det_order: int = 0,
    k_ar_diff: int = 1,
    n_sims: int = 2000,
    seed: int = 0,
    block_size: int | None = None,
) -> dict[str, float | int]:
    """One-sided p-value under correlation-preserving block-bootstrap null."""
    null = block_bootstrap_trace_null(
        log_prices,
        det_order=det_order,
        k_ar_diff=k_ar_diff,
        n_sims=n_sims,
        seed=seed,
        block_size=block_size,
    )
    raw = _pvalue_from_null(observed_trace, null, n_sims)
    t_ret = max(1, log_prices.shape[0] - 1)
    bs = block_size if block_size is not None else default_block_size(t_ret)
    return {
        "pvalue_bb": raw["pvalue"],
        "n_sims": raw["n_sims"],
        "n_valid": raw["n_valid"],
        "null_mean": raw["null_mean"],
        "null_p95": raw["null_p95"],
        "null_p99": raw["null_p99"],
        "block_size": int(bs),
    }


def dual_null_trace_pvalues(
    observed_trace: float,
    log_prices: np.ndarray,
    *,
    det_order: int = 0,
    k_ar_diff: int = 1,
    n_sims: int = 2000,
    seed: int = 0,
    block_size: int | None = None,
) -> dict[str, float | int]:
    """Compare independent-RW and block-bootstrap nulls for one observed trace."""
    n_obs, n_series = log_prices.shape
    rw = monte_carlo_trace_pvalue(
        observed_trace,
        n_obs=n_obs,
        n_series=n_series,
        det_order=det_order,
        k_ar_diff=k_ar_diff,
        n_sims=n_sims,
        seed=seed,
    )
    bb = block_bootstrap_trace_pvalue(
        observed_trace,
        log_prices,
        det_order=det_order,
        k_ar_diff=k_ar_diff,
        n_sims=n_sims,
        seed=seed + 10_000,  # separate stream
        block_size=block_size,
    )
    p_rw = float(rw["pvalue_mc"])
    p_bb = float(bb["pvalue_bb"])
    return {
        "pvalue_mc": p_rw,
        "pvalue_bb": p_bb,
        "p_ratio_bb_over_rw": (p_bb / p_rw) if p_rw > 0 else float("inf"),
        "mc_null_p95": float(rw["null_p95"]),
        "mc_null_p99": float(rw["null_p99"]),
        "bb_null_p95": float(bb["null_p95"]),
        "bb_null_p99": float(bb["null_p99"]),
        "block_size": int(bb["block_size"]),
        "n_sims": n_sims,
    }


def _parse_symbols(pipe_joined: str) -> list[str]:
    return [s for s in str(pipe_joined).split("|") if s]


def load_basket_log_prices(
    store: ParquetStore,
    symbols: list[str],
    *,
    min_obs: int,
) -> pd.DataFrame:
    frames = {sym: load_symbol_frame(store, sym) for sym in symbols}
    return _aligned_log_closes(frames, symbols, min_obs=min_obs)


def enrich_top_baskets_monte_carlo(
    df: pd.DataFrame,
    *,
    top_n: int = 4,
    n_sims: int = 2000,
    seed: int = 0,
    rank_by: str = "trace_r0",
    log_price_lookup: dict[str, np.ndarray] | None = None,
    block_bootstrap: bool = False,
    block_size: int | None = None,
    store: ParquetStore | None = None,
    min_obs: int = 180,
) -> pd.DataFrame:
    """Run MC (and optional block-bootstrap) p-values for strongest baskets.

    If ``block_bootstrap`` and a store/lookup is available, also fills
    ``pvalue_bb`` and ratio columns.
    """
    if df.empty:
        return df
    out = df.copy()
    for col in (
        "pvalue_mc",
        "pvalue_table",
        "pvalue_bb",
        "p_ratio_bb_over_rw",
        "mc_null_p95",
        "mc_null_p99",
        "bb_null_p95",
        "bb_null_p99",
    ):
        out[col] = np.nan
    out["mc_n_sims"] = 0
    out["block_size"] = pd.Series([pd.NA] * len(out), dtype="Int64")
    if "pvalue" in out.columns:
        out["pvalue_table"] = out["pvalue"]

    ranked = out.sort_values(rank_by, ascending=False).head(top_n)
    ranked_list = list(ranked.index)
    for idx, row in ranked.iterrows():
        n_series = int(row["n_symbols"])
        n_obs = int(row["n_obs"])
        observed = float(row["trace_r0"])
        det_order = int(row.get("det_order", 0))
        k_ar_diff = int(row.get("k_ar_diff", 1))
        basket_seed = seed + 17 * (1 + ranked_list.index(idx))
        syms = _parse_symbols(row["symbols"])

        table_p = approx_trace_pvalue(
            observed,
            float(row["cv90_r0"]),
            float(row["cv95_r0"]),
            float(row["cv99_r0"]),
        )
        out.at[idx, "pvalue_table"] = table_p

        log_mat: np.ndarray | None = None
        if log_price_lookup is not None and row["symbols"] in log_price_lookup:
            log_mat = log_price_lookup[row["symbols"]]
        elif block_bootstrap and store is not None:
            try:
                aligned = load_basket_log_prices(store, syms, min_obs=min_obs)
                log_mat = aligned[syms].to_numpy(dtype=float)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load prices for %s: %s", syms, exc)

        if block_bootstrap and log_mat is not None:
            logger.info(
                "Dual-null Johansen %s (n=%d dim=%d sims=%d trace=%.3f) …",
                row["symbols"],
                n_obs,
                n_series,
                n_sims,
                observed,
            )
            dual = dual_null_trace_pvalues(
                observed,
                log_mat,
                det_order=det_order,
                k_ar_diff=k_ar_diff,
                n_sims=n_sims,
                seed=basket_seed,
                block_size=block_size,
            )
            out.at[idx, "pvalue_mc"] = dual["pvalue_mc"]
            out.at[idx, "pvalue_bb"] = dual["pvalue_bb"]
            out.at[idx, "p_ratio_bb_over_rw"] = dual["p_ratio_bb_over_rw"]
            out.at[idx, "mc_null_p95"] = dual["mc_null_p95"]
            out.at[idx, "mc_null_p99"] = dual["mc_null_p99"]
            out.at[idx, "bb_null_p95"] = dual["bb_null_p95"]
            out.at[idx, "bb_null_p99"] = dual["bb_null_p99"]
            out.at[idx, "mc_n_sims"] = n_sims
            out.at[idx, "block_size"] = dual["block_size"]
            logger.info(
                "Dual %s: p_table≈%.4f  p_rw=%.4f  p_bb=%.4f  ratio=%.2f "
                "(rw p95=%.2f bb p95=%.2f obs=%.2f)",
                row["symbols"],
                table_p,
                dual["pvalue_mc"],
                dual["pvalue_bb"],
                dual["p_ratio_bb_over_rw"],
                dual["mc_null_p95"],
                dual["bb_null_p95"],
                observed,
            )
        else:
            logger.info(
                "MC Johansen %s (n=%d dim=%d sims=%d trace=%.3f) …",
                row["symbols"],
                n_obs,
                n_series,
                n_sims,
                observed,
            )
            mc = monte_carlo_trace_pvalue(
                observed,
                n_obs=n_obs,
                n_series=n_series,
                det_order=det_order,
                k_ar_diff=k_ar_diff,
                n_sims=n_sims,
                seed=basket_seed,
            )
            out.at[idx, "pvalue_mc"] = mc["pvalue_mc"]
            out.at[idx, "mc_n_sims"] = mc["n_sims"]
            out.at[idx, "mc_null_p95"] = mc["null_p95"]
            out.at[idx, "mc_null_p99"] = mc["null_p99"]
            logger.info(
                "MC %s: p_table≈%.4f  p_mc=%.4f  (null p95=%.2f p99=%.2f obs=%.2f)",
                row["symbols"],
                table_p,
                mc["pvalue_mc"],
                mc["null_p95"],
                mc["null_p99"],
                observed,
            )
    return out


def compare_dual_nulls_for_baskets(
    baskets: list[list[str]],
    config: DataConfig | None = None,
    *,
    exchange_id: str | None = None,
    n_sims: int = 2000,
    seed: int = 42,
    block_size: int | None = None,
    persist: bool = True,
    basename: str = "johansen_dual_null",
) -> pd.DataFrame:
    """Run table + indep-RW + block-bootstrap p-values for explicit baskets."""
    from pairs_trading.cointegration.johansen import johansen_basket

    config = config or load_config()
    eid = resolve_exchange_id(config, exchange_id)
    store = ParquetStore(config.storage.root, eid, config.ohlcv.timeframe)
    min_obs = config.cointegration.min_obs

    rows: list[dict] = []
    for i, syms in enumerate(baskets):
        aligned = load_basket_log_prices(store, syms, min_obs=min_obs)
        summary = johansen_basket(aligned, syms)
        log_mat = aligned[syms].to_numpy(dtype=float)
        dual = dual_null_trace_pvalues(
            float(summary["trace_r0"]),
            log_mat,
            det_order=int(summary["det_order"]),
            k_ar_diff=int(summary["k_ar_diff"]),
            n_sims=n_sims,
            seed=seed + 17 * (i + 1),
            block_size=block_size,
        )
        row = {
            **summary,
            "pvalue_table": summary["pvalue"],
            **{k: dual[k] for k in dual},
        }
        rows.append(row)
        logger.info(
            "Compare %s: table=%.4f rw=%.4f bb=%.4f ratio=%.2f",
            summary["symbols"],
            row["pvalue_table"],
            dual["pvalue_mc"],
            dual["pvalue_bb"],
            dual["p_ratio_bb_over_rw"],
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
                "kind": "johansen_dual_null",
                "n_sims": n_sims,
                "seed": seed,
                "block_size": block_size or "sqrt_T",
                "n_baskets": len(out),
            },
            basename=basename,
        )
    return out


def run_johansen_mc_on_scan(
    config: DataConfig | None = None,
    *,
    scan_kind: str = "baskets",
    top_n: int = 4,
    n_sims: int = 2000,
    seed: int = 0,
    exchange_id: str | None = None,
    persist: bool = True,
    block_bootstrap: bool = False,
    block_size: int | None = None,
) -> pd.DataFrame:
    """Load latest Johansen scan artifact, MC-enrich top baskets, persist."""
    config = config or load_config()
    eid = resolve_exchange_id(config, exchange_id)
    tf = config.ohlcv.timeframe
    root = config.storage.root / eid / "cointegration"
    if scan_kind == "bnb_free":
        path = root / f"johansen_bnb_free_random_latest_{tf}.parquet"
        basename = "johansen_bnb_free_mc"
    else:
        path = root / f"johansen_baskets_latest_{tf}.parquet"
        basename = "johansen_baskets_mc"
    if not path.is_file():
        raise FileNotFoundError(f"Missing Johansen scan artifact: {path}")

    store = ParquetStore(config.storage.root, eid, tf)
    df = pd.read_parquet(path)
    enriched = enrich_top_baskets_monte_carlo(
        df,
        top_n=top_n,
        n_sims=n_sims,
        seed=seed,
        block_bootstrap=block_bootstrap,
        block_size=block_size,
        store=store if block_bootstrap else None,
        min_obs=config.cointegration.min_obs,
    )
    mc_rows = enriched[enriched["pvalue_mc"].notna()]
    if persist and not mc_rows.empty:
        save_scan_results(
            enriched,
            root=config.storage.root,
            exchange_id=eid,
            timeframe=tf,
            mode=config.cointegration.persist_mode,
            meta={
                "kind": "johansen_monte_carlo",
                "source": str(path),
                "top_n": top_n,
                "n_sims": n_sims,
                "seed": seed,
                "block_bootstrap": block_bootstrap,
                "block_size": block_size or "sqrt_T",
            },
            basename=basename,
        )
    return enriched
