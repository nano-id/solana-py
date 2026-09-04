"""Multiple-testing corrections for pairwise cointegration screens."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


def apply_benjamini_hochberg(
    df: pd.DataFrame,
    *,
    pvalue_col: str = "pvalue",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Add BH-FDR adjusted p-values and significance flags.

    Columns added:
      - pvalue_fdr: Benjamini-Hochberg adjusted p-value
      - significant_fdr: True iff pvalue_fdr < alpha
      - significant_raw: alias of raw p < alpha (kept for comparison)
    """
    out = df.copy()
    if out.empty:
        out["pvalue_fdr"] = pd.Series(dtype=float)
        out["significant_fdr"] = pd.Series(dtype=bool)
        out["significant_raw"] = pd.Series(dtype=bool)
        return out

    pvals = out[pvalue_col].astype(float).to_numpy()
    reject, p_adj, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
    out["pvalue_fdr"] = p_adj
    out["significant_fdr"] = reject
    out["significant_raw"] = pvals < alpha
    # Keep legacy `significant` aligned with FDR for downstream filters,
    # while preserving raw via significant_raw.
    out["significant"] = out["significant_fdr"]
    return out


def apply_benjamini_yekutieli(
    df: pd.DataFrame,
    *,
    pvalue_col: str = "pvalue",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Add Benjamini–Yekutieli FDR (valid under arbitrary dependence).

    BY is stricter than BH: it multiplies the BH threshold by harmonic number
    H_m = Σ_{i=1}^m 1/i. Prefer BY when tests share factors / overlapping assets
    (Johansen baskets, hub-contaminated pair screens).

    Columns added:
      - pvalue_by: BY-adjusted p-value
      - significant_by: True iff rejected at FDR-BY level α
    """
    out = df.copy()
    if out.empty:
        out["pvalue_by"] = pd.Series(dtype=float)
        out["significant_by"] = pd.Series(dtype=bool)
        return out

    pvals = out[pvalue_col].astype(float).to_numpy()
    reject, p_adj, _, _ = multipletests(pvals, alpha=alpha, method="fdr_by")
    out["pvalue_by"] = p_adj
    out["significant_by"] = reject
    if "significant_raw" not in out.columns:
        out["significant_raw"] = pvals < alpha
    return out


def apply_bh_and_by(
    df: pd.DataFrame,
    *,
    pvalue_col: str = "pvalue",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Apply both BH and BY; keep BH columns as the primary `significant` flag."""
    out = apply_benjamini_hochberg(df, pvalue_col=pvalue_col, alpha=alpha)
    out = apply_benjamini_yekutieli(out, pvalue_col=pvalue_col, alpha=alpha)
    return out


def hub_involvement_summary(
    df: pd.DataFrame,
    *,
    hub_symbol: str,
    significant_col: str = "significant_raw",
) -> dict[str, float | int | str]:
    """Compare hit-rate for pairs involving `hub_symbol` vs the rest."""
    hub = hub_symbol.upper()
    involves = df["y_symbol"].str.upper().eq(hub) | df["x_symbol"].str.upper().eq(hub)
    n_hub = int(involves.sum())
    n_other = int((~involves).sum())
    hits_hub = int(df.loc[involves, significant_col].sum()) if n_hub else 0
    hits_other = int(df.loc[~involves, significant_col].sum()) if n_other else 0
    rate_hub = hits_hub / n_hub if n_hub else float("nan")
    rate_other = hits_other / n_other if n_other else float("nan")
    return {
        "hub_symbol": hub_symbol,
        "n_pairs_with_hub": n_hub,
        "n_hits_with_hub": hits_hub,
        "hit_rate_hub": rate_hub,
        "n_pairs_without_hub": n_other,
        "n_hits_without_hub": hits_other,
        "hit_rate_other": rate_other,
        "hit_rate_ratio": (rate_hub / rate_other) if rate_other and rate_other > 0 else float("inf"),
    }
