"""Step 3 — ADF stationarity filter on BTC-beta residuals.

Coins whose residual fails ADF at configured α never enter the candidate list.
Rule-5 remains locked; no signals / PnL / claims.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from pairs_trading.factor_beta.config import FactorBetaConfig, load_factor_beta_config
from pairs_trading.factor_beta.factor_model import fit_rolling_factor


def _adf_one(series: pd.Series, *, maxlag: int | None = None) -> dict:
    s = series.dropna()
    n = int(len(s))
    if n < 30:
        return {
            "n_obs": n,
            "adf_stat": None,
            "adf_pvalue": None,
            "usedlag": None,
            "nobs": None,
            "critical_values": None,
            "status": "insufficient_obs",
            "stationary": False,
        }
    # autolag AIC; regression='c' (constant, no trend) — residual should mean-revert to ~0
    res = adfuller(s.to_numpy(), maxlag=maxlag, regression="c", autolag="AIC")
    return {
        "n_obs": n,
        "adf_stat": float(res[0]),
        "adf_pvalue": float(res[1]),
        "usedlag": int(res[2]),
        "nobs": int(res[3]),
        "critical_values": {str(k): float(v) for k, v in res[4].items()},
        "status": "ok",
        "stationary": bool(res[1] < 0.05),  # overwritten by caller with config threshold
    }


def run_adf_filter(
    config: FactorBetaConfig | None = None,
    *,
    lookback_days: int | None = None,
) -> dict:
    """ADF-test each coin's rolling residual; emit pass/fail candidate list."""
    config = config or load_factor_beta_config()
    lookback = int(lookback_days or config.factor.beta_lookback_days_default)
    pmax = float(config.stationarity.adf_pvalue_max)
    result = fit_rolling_factor(config, lookback_days=lookback)

    per_coin: list[dict] = []
    passed: list[str] = []
    failed: list[str] = []

    for sym in result.residual.columns:
        # Use residual only where β/α valid (already NaN-masked in fit)
        e = result.residual[sym].dropna()
        adf = _adf_one(e)
        stationary = bool(
            adf["status"] == "ok"
            and adf["adf_pvalue"] is not None
            and adf["adf_pvalue"] <= pmax
        )
        adf["stationary"] = stationary
        adf["adf_pvalue_max"] = pmax
        row = {
            "symbol": sym,
            "lookback_days": lookback,
            "n_valid_residual": int(len(e)),
            "resid_std": float(e.std(ddof=0)) if len(e) else None,
            "adf": adf,
            "passes_stationarity_gate": stationary,
        }
        per_coin.append(row)
        (passed if stationary else failed).append(sym)

    # Sort by p-value ascending among tested
    per_coin_sorted = sorted(
        per_coin,
        key=lambda r: (
            r["adf"]["adf_pvalue"] is None,
            r["adf"]["adf_pvalue"] if r["adf"]["adf_pvalue"] is not None else 9.0,
        ),
    )

    return {
        "step": 3,
        "step_name": "residual_adf_stationarity_filter",
        "status": "complete",
        "interpretation_allowed": False,
        "lookback_days": lookback,
        "adf_pvalue_max": pmax,
        "adf_regression": "c",
        "adf_autolag": "AIC",
        "n_coins_tested": len(per_coin),
        "n_passed": len(passed),
        "n_failed": len(failed),
        "passed_symbols": passed,
        "failed_symbols": failed,
        "per_coin": per_coin_sorted,
        "rules_still_locked": True,
        "rule5_reminder": {
            "5a_round_trip_bps": config.costs.round_trip_bps,
            "5b_min_edge_to_cost": config.entry_gate.min_edge_to_cost_multiple,
            "5c_min_windows": config.evaluation.min_independent_windows,
            "5d_dual_null": config.multiple_testing.dual_null,
            "5d_fdr": config.multiple_testing.fdr_method,
            "5e_max_abs_resid_corr": config.common_factor_control.max_abs_residual_corr,
            "note": (
                "ADF pass ≠ tradeable. Candidates still face dual-null+BY (5d), "
                "residual-corr clustering (5e), cost gate (5a/5b), multi-window (5c)."
            ),
        },
        "note": (
            "Step 3 only filters non-stationary residuals out of the candidate list. "
            "No z-signals, no PnL, no 'edge exists' claim. "
            "Effective independent-signal count still unknown until Step 5e clustering."
        ),
        "next_step": 4,
        "next_step_name": "residual_zscore_signal_skeleton",
    }


def save_adf_report(report: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "step3_adf_stationarity_filter.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    # also write candidate list alone for downstream steps
    cand = {
        "lookback_days": report["lookback_days"],
        "adf_pvalue_max": report["adf_pvalue_max"],
        "passed_symbols": report["passed_symbols"],
        "failed_symbols": report["failed_symbols"],
    }
    (output_dir / "step3_candidates.json").write_text(
        json.dumps(cand, indent=2), encoding="utf-8"
    )
    return path
