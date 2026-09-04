"""Smoke tests for rolling window helpers and Johansen wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pairs_trading.cointegration.fdr import apply_benjamini_hochberg
from pairs_trading.cointegration.hf_retest import timeframe_hours
from pairs_trading.cointegration.johansen import approx_trace_pvalue, johansen_basket
from pairs_trading.cointegration.rolling import iter_windows, summarize_rolling


def test_iter_windows_steps():
    # 400 days span, 180d window, 90d step → several windows
    start = 1_700_000_000_000
    end = start + 400 * 86_400_000
    wins = iter_windows(start, end, window_days=180, step_days=90)
    assert len(wins) >= 2
    assert all(w[1] - w[0] == 180 * 86_400_000 for w in wins)


def test_summarize_rolling_counts():
    df = pd.DataFrame(
        {
            "window_start": ["2023-01-01", "2023-01-01", "2023-04-01"],
            "window_end": ["2023-07-01", "2023-07-01", "2023-10-01"],
            "window_days": [180, 180, 180],
            "pair": ["A|B", "C|D", "A|B"],
            "significant_raw": [True, False, True],
            "significant_fdr": [True, False, False],
        }
    )
    s = summarize_rolling(df)
    assert len(s) == 2
    row0 = s.iloc[0]
    assert row0["n_pairs"] == 2
    assert row0["n_raw"] == 1
    assert row0["n_fdr"] == 1


def test_johansen_basket_on_cointegrated_synthetic():
    rng = np.random.default_rng(0)
    n = 500
    f = np.cumsum(rng.normal(0, 0.01, size=n))
    # Three series sharing one common stochastic trend → expect ≥1 relation.
    y1 = f + rng.normal(0, 0.01, size=n)
    y2 = 0.5 * f + rng.normal(0, 0.01, size=n)
    y3 = -0.3 * f + rng.normal(0, 0.01, size=n)
    log_prices = pd.DataFrame({"A": y1, "B": y2, "C": y3})
    out = johansen_basket(log_prices, ["A", "B", "C"], det_order=0, k_ar_diff=1)
    assert out["n_symbols"] == 3
    assert out["has_coint_95"] is True
    assert "pvalue" in out
    assert out["pvalue"] < 0.05


def test_approx_trace_pvalue_monotonic():
    cv90, cv95, cv99 = 20.0, 25.0, 30.0
    assert approx_trace_pvalue(0.0, cv90, cv95, cv99) > 0.5
    assert approx_trace_pvalue(cv90, cv90, cv95, cv99) == 0.10
    assert approx_trace_pvalue(cv95, cv90, cv95, cv99) == 0.05
    assert approx_trace_pvalue(cv99, cv90, cv95, cv99) == 0.005
    assert approx_trace_pvalue(cv95 + 1, cv90, cv95, cv99) < 0.05


def test_johansen_fdr_column_roundtrip():
    df = pd.DataFrame(
        {
            "symbols": ["A|B|C", "D|E|F", "G|H|I"],
            "pvalue": [0.01, 0.04, 0.20],
            "has_coint_95": [True, True, False],
        }
    )
    out = apply_benjamini_hochberg(df, pvalue_col="pvalue", alpha=0.05)
    assert "significant_fdr" in out.columns
    assert "pvalue_fdr" in out.columns
    assert bool(out.loc[out["pvalue"] == 0.01, "significant_fdr"].iloc[0])


def test_timeframe_hours():
    assert timeframe_hours("4h") == 4.0
    assert timeframe_hours("1d") == 24.0
