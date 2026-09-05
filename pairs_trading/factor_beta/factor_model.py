"""Step 2 — rolling BTC-beta factor model: log(coin)=α_t+β_t·log(BTC)+ε_t."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pairs_trading.factor_beta.config import FactorBetaConfig, load_factor_beta_config
from pairs_trading.factor_beta.data import build_log_price_panel


@dataclass
class RollingFactorResult:
    lookback_days: int
    alpha: pd.DataFrame  # columns = coins
    beta: pd.DataFrame
    residual: pd.DataFrame
    r2: pd.DataFrame
    n_valid_by_symbol: dict[str, int]


def rolling_ols_alpha_beta_resid(
    y: pd.Series,
    x: pd.Series,
    *,
    window: int,
    min_obs: int,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Rolling OLS y = α + β x + ε via rolling moments (no look-ahead).

    At time t, uses observations [t-window+1, t] inclusive (pandas rolling default).
    """
    if window < min_obs:
        raise ValueError(f"window {window} < min_obs {min_obs}")
    df = pd.DataFrame({"y": y.astype(float), "x": x.astype(float)}).dropna()
    # Precompute products, then rolling sums (Rolling has no .pow)
    df = df.assign(
        xx=df["x"] * df["x"],
        yy=df["y"] * df["y"],
        xy=df["x"] * df["y"],
    )
    roll = df.rolling(window=window, min_periods=min_obs)
    n = roll["y"].count()
    sum_x = roll["x"].sum()
    sum_y = roll["y"].sum()
    sum_xx = roll["xx"].sum()
    sum_yy = roll["yy"].sum()
    sum_xy = roll["xy"].sum()

    # β = (n·Σxy − Σx·Σy) / (n·Σxx − (Σx)^2)
    denom = n * sum_xx - sum_x * sum_x
    numer = n * sum_xy - sum_x * sum_y
    beta = numer / denom
    alpha = (sum_y - beta * sum_x) / n
    # residual at t uses α_t, β_t estimated on window ending at t (includes t)
    resid = df["y"] - alpha - beta * df["x"]
    # R² on the window (optional diagnostic)
    ss_tot = sum_yy - (sum_y * sum_y) / n
    ss_res = (
        sum_yy
        - 2 * alpha * sum_y
        - 2 * beta * sum_xy
        + n * alpha * alpha
        + 2 * alpha * beta * sum_x
        + beta * beta * sum_xx
    )
    r2 = 1.0 - ss_res / ss_tot
    # Invalidate degenerate windows
    bad = denom.abs() < 1e-12
    alpha = alpha.mask(bad)
    beta = beta.mask(bad)
    resid = resid.mask(bad | alpha.isna())
    r2 = r2.mask(bad | ss_tot.abs() < 1e-12)
    return alpha, beta, resid, r2


def fit_rolling_factor(
    config: FactorBetaConfig | None = None,
    *,
    lookback_days: int | None = None,
) -> RollingFactorResult:
    config = config or load_factor_beta_config()
    lookback = int(lookback_days or config.factor.beta_lookback_days_default)
    min_obs = int(config.factor.min_obs)
    if lookback < min_obs:
        raise ValueError(f"lookback_days {lookback} < min_obs {min_obs}")

    panel = build_log_price_panel(config)
    lp = panel.log_prices
    factor = panel.factor_symbol
    x = lp[factor]
    coins = [c for c in lp.columns if c != factor]

    alphas: dict[str, pd.Series] = {}
    betas: dict[str, pd.Series] = {}
    resids: dict[str, pd.Series] = {}
    r2s: dict[str, pd.Series] = {}
    n_valid: dict[str, int] = {}

    for coin in coins:
        a, b, e, r2 = rolling_ols_alpha_beta_resid(
            lp[coin], x, window=lookback, min_obs=min_obs
        )
        alphas[coin] = a
        betas[coin] = b
        resids[coin] = e
        r2s[coin] = r2
        n_valid[coin] = int(e.notna().sum())

    return RollingFactorResult(
        lookback_days=lookback,
        alpha=pd.DataFrame(alphas).sort_index(),
        beta=pd.DataFrame(betas).sort_index(),
        residual=pd.DataFrame(resids).sort_index(),
        r2=pd.DataFrame(r2s).sort_index(),
        n_valid_by_symbol=n_valid,
    )


def factor_model_report(
    config: FactorBetaConfig | None = None,
    *,
    lookback_days: int | None = None,
) -> dict:
    """Step-2 report: rolling β stats only. No trading interpretation."""
    config = config or load_factor_beta_config()
    lookback = int(lookback_days or config.factor.beta_lookback_days_default)
    result = fit_rolling_factor(config, lookback_days=lookback)

    per_coin: list[dict] = []
    for coin in result.beta.columns:
        b = result.beta[coin].dropna()
        a = result.alpha[coin].dropna()
        r2 = result.r2[coin].dropna()
        e = result.residual[coin].dropna()
        per_coin.append(
            {
                "symbol": coin,
                "n_valid_residual": int(len(e)),
                "beta_last": float(b.iloc[-1]) if len(b) else None,
                "beta_mean": float(b.mean()) if len(b) else None,
                "beta_std": float(b.std(ddof=0)) if len(b) else None,
                "alpha_last": float(a.iloc[-1]) if len(a) else None,
                "r2_mean": float(r2.mean()) if len(r2) else None,
                "resid_std": float(e.std(ddof=0)) if len(e) else None,
            }
        )

    # Residual pairwise |corr| snapshot (Rule 5e preview only — not a verdict)
    resid = result.residual.dropna(how="any")
    corr_preview = None
    if resid.shape[0] >= 30 and resid.shape[1] >= 2:
        cmat = resid.corr().abs()
        # upper triangle pairs above threshold
        pairs = []
        cols = list(cmat.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                v = float(cmat.iloc[i, j])
                if v > config.common_factor_control.max_abs_residual_corr:
                    pairs.append({"a": cols[i], "b": cols[j], "abs_corr": round(v, 4)})
        corr_preview = {
            "threshold": config.common_factor_control.max_abs_residual_corr,
            "n_pairs_above": len(pairs),
            "pairs_above": pairs[:30],  # cap for readability
            "note": "Preview only — Rule 5e clustering applied in later step, not a pass/fail here.",
        }

    return {
        "step": 2,
        "step_name": "factor_model_rolling_beta",
        "status": "complete",
        "interpretation_allowed": False,
        "timescales": {
            "beta_lookback_days": lookback,
            "beta_lookback_days_grid": list(config.factor.beta_lookback_days_grid),
            "beta_lookback_days_default": config.factor.beta_lookback_days_default,
            "trade_eval_window_weeks": [
                config.evaluation.window_weeks_min,
                config.evaluation.window_weeks_max,
            ],
            "zscore_window_grid_days": list(config.signal.zscore_window_grid),
            "note": (
                "THREE independent timescales — do not mix: "
                "(1) beta_lookback_days for OLS α,β; "
                "(2) evaluation.window_weeks for Rule-5c trade panels; "
                "(3) signal.zscore_window_grid for residual z."
            ),
        },
        "model": "log(coin)_t = α_t + β_t·log(BTC)_t + ε_t  (rolling OLS, window ending at t)",
        "factor_symbol": config.factor.symbol,
        "n_coins": len(result.beta.columns),
        "aligned_residual_start": (
            result.residual.dropna(how="all").index.min().date().isoformat()
            if result.residual.notna().any().any()
            else None
        ),
        "aligned_residual_end": (
            result.residual.dropna(how="all").index.max().date().isoformat()
            if result.residual.notna().any().any()
            else None
        ),
        "per_coin": per_coin,
        "rule5e_residual_corr_preview": corr_preview,
        "rules_still_locked": True,
        "next_step": 3,
        "next_step_name": "residual_adf_stationarity_filter",
        "note": (
            "Step 2 fits rolling α,β,ε only. "
            "No entry signals, no PnL, no 'edge exists' claim."
        ),
    }


def save_factor_outputs(
    result: RollingFactorResult,
    report: dict,
    output_dir: Path,
    *,
    write_canonical: bool = False,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "report": str(output_dir / f"step2_factor_model_report_lb{result.lookback_days}.json"),
        "beta": str(output_dir / f"step2_beta_lookback{result.lookback_days}.parquet"),
        "alpha": str(output_dir / f"step2_alpha_lookback{result.lookback_days}.parquet"),
        "residual": str(
            output_dir / f"step2_residual_lookback{result.lookback_days}.parquet"
        ),
    }
    Path(paths["report"]).write_text(json.dumps(report, indent=2), encoding="utf-8")
    result.beta.to_parquet(paths["beta"])
    result.alpha.to_parquet(paths["alpha"])
    result.residual.to_parquet(paths["residual"])
    if write_canonical:
        canon = output_dir / "step2_factor_model_report.json"
        canon.write_text(json.dumps(report, indent=2), encoding="utf-8")
        paths["canonical_report"] = str(canon)
    return paths


def run_step2_all_lookbacks(config: FactorBetaConfig | None = None) -> dict:
    """Fit each grid lookback; write default lookback series + summary for all."""
    config = config or load_factor_beta_config()
    out_dir = Path(config.storage.output_dir)
    summaries = []
    default = config.factor.beta_lookback_days_default
    default_paths = None
    default_report = None
    for lb in config.factor.beta_lookback_days_grid:
        result = fit_rolling_factor(config, lookback_days=lb)
        report = factor_model_report(config, lookback_days=lb)
        paths = save_factor_outputs(
            result, report, out_dir, write_canonical=(lb == default)
        )
        beta_means = [c["beta_mean"] for c in report["per_coin"] if c["beta_mean"] is not None]
        r2_means = [c["r2_mean"] for c in report["per_coin"] if c["r2_mean"] is not None]
        summaries.append(
            {
                "lookback_days": lb,
                "is_default": lb == default,
                "n_coins": report["n_coins"],
                "median_beta_mean": float(np.median(beta_means)) if beta_means else None,
                "median_r2_mean": float(np.median(r2_means)) if r2_means else None,
                "rule5e_pairs_above": (
                    report["rule5e_residual_corr_preview"]["n_pairs_above"]
                    if report["rule5e_residual_corr_preview"]
                    else None
                ),
                "paths": paths,
            }
        )
        if lb == default:
            default_paths = paths
            default_report = report
    bundle = {
        "step": 2,
        "status": "complete",
        "interpretation_allowed": False,
        "default_lookback_days": default,
        "grid_summaries": summaries,
        "default_paths": default_paths,
        "default_report": default_report,
    }
    (out_dir / "step2_grid_summary.json").write_text(
        json.dumps({k: v for k, v in bundle.items() if k != "default_report"}, indent=2),
        encoding="utf-8",
    )
    return bundle
