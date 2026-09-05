"""Ornstein-Uhlenbeck / AR(1) half-life of a mean-reverting spread."""

from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_half_life(spread: pd.Series | np.ndarray) -> float:
    """Estimate mean-reversion half-life in bars via AR(1).

    Fit: spread_t = φ * spread_{t-1} + ε
    half_life = -ln(2) / ln(φ)   for 0 < φ < 1

    Returns +inf when the series is not mean-reverting (φ <= 0 or φ >= 1).
    """
    s = pd.Series(spread).astype(float).dropna()
    if len(s) < 3:
        return float("inf")

    lag = s.shift(1).iloc[1:]
    now = s.iloc[1:]
    # OLS through origin is wrong for OU; include intercept via demean or const.
    # Standard pairs-trading half-life uses: Δy = θ * y_{t-1} + ε  (with const),
    # half_life = -ln(2)/θ. Equivalent AR(1): y_t = φ y_{t-1} + c + ε, φ=1+θ.
    x = np.column_stack([np.ones(len(lag)), lag.to_numpy()])
    y = now.to_numpy()
    try:
        beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    except np.linalg.LinAlgError:
        return float("inf")

    phi = float(beta[1])
    if phi <= 0.0 or phi >= 1.0:
        return float("inf")
    return float(-np.log(2.0) / np.log(phi))
