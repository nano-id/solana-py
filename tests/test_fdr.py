"""Tests for Benjamini-Hochberg FDR helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pairs_trading.cointegration.fdr import apply_benjamini_hochberg, hub_involvement_summary


def test_bh_fdr_reduces_false_discoveries():
    # 100 null p-values ~ U(0,1); a few tiny ones. Raw α=0.05 expects ~5 hits;
    # BH should be stricter / fewer or equal.
    rng = np.random.default_rng(0)
    pvals = np.concatenate([rng.uniform(0, 1, size=95), [1e-6, 1e-5, 1e-4, 0.001, 0.01]])
    df = pd.DataFrame(
        {
            "y_symbol": [f"Y{i}" for i in range(len(pvals))],
            "x_symbol": [f"X{i}" for i in range(len(pvals))],
            "pvalue": pvals,
        }
    )
    out = apply_benjamini_hochberg(df, alpha=0.05)
    assert "pvalue_fdr" in out.columns
    assert out["significant_raw"].sum() >= out["significant_fdr"].sum()
    # The tiniest p should survive FDR.
    assert bool(out.loc[out["pvalue"].idxmin(), "significant_fdr"])


def test_hub_involvement_summary():
    df = pd.DataFrame(
        {
            "y_symbol": ["BNB/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT"],
            "x_symbol": ["ETH/USDT", "XRP/USDT", "UNI/USDT", "DOT/USDT"],
            "significant_raw": [True, True, True, False],
            "significant_fdr": [True, False, False, False],
        }
    )
    raw = hub_involvement_summary(df, hub_symbol="BNB/USDT", significant_col="significant_raw")
    assert raw["n_pairs_with_hub"] == 2
    assert raw["n_hits_with_hub"] == 2
    assert raw["hit_rate_hub"] == 1.0
    assert raw["n_hits_without_hub"] == 1
