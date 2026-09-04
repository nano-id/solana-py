"""Monte Carlo null p-values for Johansen trace (H0: r=0)."""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from pairs_trading.cointegration.johansen import approx_trace_pvalue
from pairs_trading.cointegration.persist import save_scan_results
from pairs_trading.cointegration.scanner import resolve_exchange_id
from pairs_trading.data.config import DataConfig, load_config

logger = logging.getLogger(__name__)


def _trace_r0(
    data: np.ndarray,
    *,
    det_order: int,
    k_ar_diff: int,
) -> float:
    res = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)
    return float(np.asarray(res.lr1, dtype=float)[0])


def simulate_independent_random_walks(
    n_obs: int,
    n_series: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """I(1) series with no cointegration: independent Gaussian random walks."""
    shocks = rng.normal(size=(n_obs, n_series))
    return np.cumsum(shocks, axis=0)


def monte_carlo_trace_null(
    *,
    n_obs: int,
    n_series: int,
    det_order: int = 0,
    k_ar_diff: int = 1,
    n_sims: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    """Draw null distribution of Johansen trace statistic under r=0."""
    rng = np.random.default_rng(seed)
    out = np.empty(n_sims, dtype=float)
    for i in range(n_sims):
        walks = simulate_independent_random_walks(n_obs, n_series, rng)
        try:
            out[i] = _trace_r0(walks, det_order=det_order, k_ar_diff=k_ar_diff)
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
    """One-sided MC p-value: P(trace_null >= observed | H0: r=0).

    Uses (1 + #{sim ≥ obs}) / (N + 1) (Davison & Hinkley).
    """
    null = monte_carlo_trace_null(
        n_obs=n_obs,
        n_series=n_series,
        det_order=det_order,
        k_ar_diff=k_ar_diff,
        n_sims=n_sims,
        seed=seed,
    )
    valid = null[np.isfinite(null)]
    n_valid = int(len(valid))
    if n_valid == 0 or not np.isfinite(observed_trace):
        return {
            "pvalue_mc": 1.0,
            "n_sims": n_sims,
            "n_valid": n_valid,
            "null_mean": float("nan"),
            "null_p95": float("nan"),
            "null_p99": float("nan"),
        }
    n_ge = int(np.sum(valid >= observed_trace))
    p_mc = (1.0 + n_ge) / (n_valid + 1.0)
    return {
        "pvalue_mc": float(p_mc),
        "n_sims": n_sims,
        "n_valid": n_valid,
        "null_mean": float(np.mean(valid)),
        "null_p95": float(np.quantile(valid, 0.95)),
        "null_p99": float(np.quantile(valid, 0.99)),
    }


def enrich_top_baskets_monte_carlo(
    df: pd.DataFrame,
    *,
    top_n: int = 4,
    n_sims: int = 2000,
    seed: int = 0,
    rank_by: str = "trace_r0",
) -> pd.DataFrame:
    """Run MC p-values for the strongest `top_n` baskets; leave others NaN.

    Compares table-interpolated `pvalue` vs Monte Carlo `pvalue_mc`.
    """
    if df.empty:
        return df
    out = df.copy()
    out["pvalue_mc"] = np.nan
    out["pvalue_table"] = out["pvalue"] if "pvalue" in out.columns else np.nan
    out["mc_n_sims"] = 0
    out["mc_null_p95"] = np.nan
    out["mc_null_p99"] = np.nan

    # Prefer highest trace (strongest evidence against r=0)
    ranked = out.sort_values(rank_by, ascending=False).head(top_n)
    for idx, row in ranked.iterrows():
        n_series = int(row["n_symbols"])
        n_obs = int(row["n_obs"])
        observed = float(row["trace_r0"])
        det_order = int(row.get("det_order", 0))
        k_ar_diff = int(row.get("k_ar_diff", 1))
        # Distinct seed per basket so null draws are independent across top-N
        basket_seed = seed + 17 * (1 + list(ranked.index).index(idx))
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
        # Table p for the same observed (already on row); also recompute for clarity
        table_p = approx_trace_pvalue(
            observed,
            float(row["cv90_r0"]),
            float(row["cv95_r0"]),
            float(row["cv99_r0"]),
        )
        out.at[idx, "pvalue_mc"] = mc["pvalue_mc"]
        out.at[idx, "pvalue_table"] = table_p
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


def run_johansen_mc_on_scan(
    config: DataConfig | None = None,
    *,
    scan_kind: str = "baskets",
    top_n: int = 4,
    n_sims: int = 2000,
    seed: int = 0,
    exchange_id: str | None = None,
    persist: bool = True,
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

    df = pd.read_parquet(path)
    enriched = enrich_top_baskets_monte_carlo(
        df, top_n=top_n, n_sims=n_sims, seed=seed
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
            },
            basename=basename,
        )
    return enriched
