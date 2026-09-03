"""Factor-neutral preprocessing: regress log(coin) on log(factor), keep residual."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)


def _log_close(ohlcv: pd.DataFrame) -> pd.DataFrame:
    out = ohlcv[["timestamp", "close"]].copy()
    close = out["close"].astype(float)
    if (close <= 0).any():
        raise ValueError("Non-positive close cannot be log-transformed")
    out["log_close"] = np.log(close)
    return out[["timestamp", "log_close"]]


def residualize_against_factor(
    coin_ohlcv: pd.DataFrame,
    factor_ohlcv: pd.DataFrame,
    *,
    min_obs: int,
) -> tuple[pd.DataFrame, float, float, int]:
    """OLS: log(coin) = α + β·log(factor) + ε. Return residual as OHLCV-like frame.

    The returned frame uses `close` = residual (already in log-space). Callers must
    NOT log-transform again (use price_space='level' in Engle-Granger).
    """
    left = _log_close(coin_ohlcv).rename(columns={"log_close": "y"})
    right = _log_close(factor_ohlcv).rename(columns={"log_close": "f"})
    merged = left.merge(right, on="timestamp", how="inner").sort_values("timestamp")
    merged = merged.dropna()
    if len(merged) < min_obs:
        raise ValueError(
            f"Insufficient overlap for residualization: {len(merged)} < {min_obs}"
        )

    x = sm.add_constant(merged["f"].to_numpy(), has_constant="add")
    model = sm.OLS(merged["y"].to_numpy(), x).fit()
    alpha = float(model.params[0])
    beta = float(model.params[1])
    resid = pd.Series(model.resid, index=merged.index)

    # Synthetic OHLCV: only timestamp/close used by align_close_prices.
    frame = pd.DataFrame(
        {
            "timestamp": merged["timestamp"].to_numpy(),
            "open": resid.to_numpy(),
            "high": resid.to_numpy(),
            "low": resid.to_numpy(),
            "close": resid.to_numpy(),
            "volume": np.ones(len(resid)),
        }
    )
    return frame, alpha, beta, len(frame)


def build_factor_residual_frames(
    frames: dict[str, pd.DataFrame],
    *,
    factor_symbol: str,
    min_obs: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, float | int]]]:
    """Residualize every non-factor symbol against `factor_symbol`.

    Factor itself is omitted from the output (residual vs self is ~0 / meaningless).
    """
    if factor_symbol not in frames:
        raise FileNotFoundError(
            f"Factor symbol {factor_symbol} missing from loaded frames"
        )
    factor = frames[factor_symbol]
    out: dict[str, pd.DataFrame] = {}
    meta: dict[str, dict[str, float | int]] = {}
    for sym, ohlcv in frames.items():
        if sym == factor_symbol:
            continue
        try:
            resid, alpha, beta, n = residualize_against_factor(
                ohlcv, factor, min_obs=min_obs
            )
        except ValueError as exc:
            logger.warning("Skip residualizing %s: %s", sym, exc)
            continue
        out[sym] = resid
        meta[sym] = {"alpha": alpha, "beta_vs_factor": beta, "n_obs": n}
        logger.info(
            "Residualized %s vs %s: β_factor=%.4f n=%d",
            sym,
            factor_symbol,
            beta,
            n,
        )
    return out, meta
