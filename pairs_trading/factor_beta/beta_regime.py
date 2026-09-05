"""Parallel diagnostic: β sign flips for selected coins (WLD/NEAR).

Not a trading step — explains whether β_last sign change looks like noise
(R² collapse) or a structural break (R² holds). Rule-5 unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pairs_trading.data.storage import ParquetStore
from pairs_trading.factor_beta.config import FactorBetaConfig, load_factor_beta_config
from pairs_trading.factor_beta.factor_model import fit_rolling_factor


def _btc_realized_vol(config: FactorBetaConfig, window: int = 30) -> pd.Series:
    store = ParquetStore(
        Path(config.storage.root),
        config.storage.exchange_id,
        config.storage.timeframe,
    )
    df = store.load(config.factor.symbol)
    ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    close = pd.Series(df["close"].astype(float).to_numpy(), index=ts).sort_index()
    close = close[~close.index.duplicated(keep="last")]
    log_ret = np.log(close).diff()
    # annualized-ish daily vol proxy: rolling std of log returns (not annualized)
    return log_ret.rolling(window=window, min_periods=window).std(ddof=0)


def _first_pos_to_neg_cross(beta: pd.Series) -> pd.Timestamp | None:
    """First date where β_t < 0 after a prior β > 0 in the valid sample."""
    b = beta.dropna()
    if b.empty:
        return None
    prev_pos = False
    for t, v in b.items():
        if v > 0:
            prev_pos = True
        elif v < 0 and prev_pos:
            return pd.Timestamp(t)
    return None


def _last_pos_to_neg_cross(beta: pd.Series) -> pd.Timestamp | None:
    """Most recent pos→neg crossing (often the regime flip of interest)."""
    b = beta.dropna()
    if b.empty:
        return None
    last = None
    prev_pos = False
    for t, v in b.items():
        if v > 0:
            prev_pos = True
        elif v < 0 and prev_pos:
            last = pd.Timestamp(t)
            prev_pos = False  # require recovering to pos before another count
    return last


def analyze_beta_sign_flip(
    config: FactorBetaConfig | None = None,
    *,
    symbols: list[str] | None = None,
    lookback_days: int = 182,
    pre_post_days: int = 30,
) -> dict:
    """Compare R² and BTC vol around β sign flips for named symbols."""
    config = config or load_factor_beta_config()
    symbols = symbols or ["WLD/USDT", "NEAR/USDT"]
    result = fit_rolling_factor(config, lookback_days=lookback_days)
    btc_vol = _btc_realized_vol(config, window=30)

    # Persist R² for audit
    out_dir = Path(config.storage.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.r2.to_parquet(out_dir / f"step2_r2_lookback{lookback_days}.parquet")

    per_symbol: list[dict] = []
    for sym in symbols:
        if sym not in result.beta.columns:
            per_symbol.append({"symbol": sym, "status": "missing_from_panel"})
            continue
        beta = result.beta[sym]
        r2 = result.r2[sym]
        first_cross = _first_pos_to_neg_cross(beta)
        last_cross = _last_pos_to_neg_cross(beta)
        cross = last_cross or first_cross

        row: dict = {
            "symbol": sym,
            "lookback_days": lookback_days,
            "beta_mean": float(beta.dropna().mean()) if beta.notna().any() else None,
            "beta_last": float(beta.dropna().iloc[-1]) if beta.notna().any() else None,
            "r2_mean": float(r2.dropna().mean()) if r2.notna().any() else None,
            "r2_last": float(r2.dropna().iloc[-1]) if r2.notna().any() else None,
            "first_pos_to_neg_date": first_cross.date().isoformat() if first_cross else None,
            "last_pos_to_neg_date": last_cross.date().isoformat() if last_cross else None,
            "analyzed_cross_date": cross.date().isoformat() if cross else None,
        }

        if cross is None:
            row["classification"] = "no_pos_to_neg_cross_found"
            row["note"] = "β never crossed from + to − in valid sample (or never positive)."
            per_symbol.append(row)
            continue

        # Windows around cross (exclude the cross day itself for pre; include post start)
        pre = slice(cross - pd.Timedelta(days=pre_post_days), cross - pd.Timedelta(days=1))
        post = slice(cross, cross + pd.Timedelta(days=pre_post_days))
        beta_pre = beta.loc[pre].dropna()
        beta_post = beta.loc[post].dropna()
        r2_pre = r2.loc[pre].dropna()
        r2_post = r2.loc[post].dropna()
        vol_pre = btc_vol.loc[pre].dropna()
        vol_post = btc_vol.loc[post].dropna()

        r2_pre_m = float(r2_pre.mean()) if len(r2_pre) else None
        r2_post_m = float(r2_post.mean()) if len(r2_post) else None
        vol_pre_m = float(vol_pre.mean()) if len(vol_pre) else None
        vol_post_m = float(vol_post.mean()) if len(vol_post) else None

        # Classification heuristic (documented, not a trade rule)
        if r2_pre_m is None or r2_post_m is None:
            classification = "insufficient_r2_sample"
            rationale = "Not enough R² observations around the cross."
        else:
            r2_drop = r2_pre_m - r2_post_m
            vol_jump = (
                (vol_post_m / vol_pre_m - 1.0)
                if (vol_pre_m and vol_post_m and vol_pre_m > 0)
                else None
            )
            # Noise: R² collapses around flip (model unreliable)
            # Structural: R² stays relatively intact while sign flips
            if r2_post_m < 0.15 or (r2_pre_m > 0 and r2_drop / max(r2_pre_m, 1e-6) > 0.40):
                classification = "likely_noisy_estimate"
                rationale = (
                    "R² weak/collapsing around sign flip → β sign change "
                    "looks like unstable OLS, not a clean regime break."
                )
            elif r2_post_m >= 0.25 and r2_drop / max(r2_pre_m, 1e-6) <= 0.25:
                classification = "possible_structural_break"
                rationale = (
                    "R² remains material and does not collapse while β sign flips "
                    "→ consistent with a real regime shift (still not a trade signal)."
                )
            else:
                classification = "ambiguous"
                rationale = (
                    "Mixed R² behavior around flip; neither clean noise nor clean break."
                )

            row["r2_drop"] = r2_drop
            row["btc_vol_jump_frac"] = vol_jump

        row.update(
            {
                "pre_post_days": pre_post_days,
                "beta_pre_mean": float(beta_pre.mean()) if len(beta_pre) else None,
                "beta_post_mean": float(beta_post.mean()) if len(beta_post) else None,
                "r2_pre_mean": r2_pre_m,
                "r2_post_mean": r2_post_m,
                "btc_vol30_pre_mean": vol_pre_m,
                "btc_vol30_post_mean": vol_post_m,
                "classification": classification,
                "rationale": rationale,
                # Sparse time series for paste report (weekly samples around cross ±60d)
                "beta_series_around_cross": _sparse_series(
                    beta, cross, pad_days=60, step_days=7
                ),
                "r2_series_around_cross": _sparse_series(
                    r2, cross, pad_days=60, step_days=7
                ),
            }
        )
        per_symbol.append(row)

    return {
        "step": "2b",
        "step_name": "beta_sign_flip_diagnostic",
        "status": "complete",
        "interpretation_allowed": False,
        "lookback_days": lookback_days,
        "symbols": symbols,
        "per_symbol": per_symbol,
        "rules_still_locked": True,
        "note": (
            "Diagnostic only. Classifications are heuristics for research hygiene, "
            "not entry/exit rules. Does not unlock trading claims."
        ),
        "next_step": 3,
        "next_step_name": "residual_adf_stationarity_filter",
    }


def _sparse_series(
    s: pd.Series,
    center: pd.Timestamp,
    *,
    pad_days: int,
    step_days: int,
) -> list[dict]:
    lo = center - pd.Timedelta(days=pad_days)
    hi = center + pd.Timedelta(days=pad_days)
    window = s.loc[lo:hi].dropna()
    if window.empty:
        return []
    # resample-ish: take points every step_days
    out = []
    last_kept = None
    for t, v in window.items():
        ts = pd.Timestamp(t)
        if last_kept is None or (ts - last_kept).days >= step_days:
            out.append({"date": ts.date().isoformat(), "value": round(float(v), 4)})
            last_kept = ts
    return out


def save_beta_regime_report(report: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "step2b_beta_sign_flip_diagnostic.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
