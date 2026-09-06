"""Engle-Granger cointegration test + hedge ratio for a single pair.

Default price space is **log**: log(y) = α + β·log(x) + ε,
spread = log(y) − β·log(x) − α. Raw (level) prices are opt-in only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

from pairs_trading.cointegration.half_life import estimate_half_life

PriceSpace = Literal["log", "level"]


@dataclass(frozen=True)
class CointegrationHit:
    y_symbol: str
    x_symbol: str
    pvalue: float
    test_statistic: float
    critical_value_5pct: float | None
    hedge_ratio: float  # β in f(y) = α + β f(x) + ε
    intercept: float
    half_life: float
    n_obs: int
    price_space: PriceSpace = "log"


def align_close_prices(
    y: pd.DataFrame,
    x: pd.DataFrame,
    *,
    min_obs: int,
    price_space: PriceSpace = "log",
) -> tuple[pd.Series, pd.Series]:
    """Inner-join on timestamp; transform closes per price_space (no look-ahead)."""
    left = y[["timestamp", "close"]].rename(columns={"close": "y"})
    right = x[["timestamp", "close"]].rename(columns={"close": "x"})
    merged = left.merge(right, on="timestamp", how="inner").sort_values("timestamp")
    merged = merged.dropna()
    if len(merged) < min_obs:
        raise ValueError(
            f"Insufficient overlapping bars: {len(merged)} < min_obs={min_obs}"
        )
    y_s = merged["y"].astype(float)
    x_s = merged["x"].astype(float)
    if price_space == "log":
        if (y_s <= 0).any() or (x_s <= 0).any():
            raise ValueError("Non-positive prices cannot be log-transformed")
        y_s = np.log(y_s)
        x_s = np.log(x_s)
    elif price_space != "level":
        raise ValueError(f"Unknown price_space={price_space!r}; use 'log' or 'level'")
    return y_s, x_s


def ols_hedge_ratio(y: pd.Series, x: pd.Series) -> tuple[float, float, pd.Series]:
    """OLS: y = α + β x + ε. Returns (β, α, residual spread)."""
    x_const = sm.add_constant(x.to_numpy(), has_constant="add")
    model = sm.OLS(y.to_numpy(), x_const).fit()
    intercept = float(model.params[0])
    beta = float(model.params[1])
    spread = pd.Series(model.resid, index=y.index, name="spread")
    return beta, intercept, spread


def evaluate_pair(
    y_symbol: str,
    x_symbol: str,
    y_ohlcv: pd.DataFrame,
    x_ohlcv: pd.DataFrame,
    *,
    min_obs: int = 100,
    autolag: str | None = "aic",
    price_space: PriceSpace = "log",
) -> CointegrationHit:
    """Engle-Granger + OLS β + half-life; always returns metrics (no p-gate)."""
    y, x = align_close_prices(
        y_ohlcv, x_ohlcv, min_obs=min_obs, price_space=price_space
    )
    score, pvalue, crit = coint(y.to_numpy(), x.to_numpy(), autolag=autolag)
    crit_5 = float(crit[1]) if crit is not None and len(crit) > 1 else None
    beta, intercept, spread = ols_hedge_ratio(y, x)
    return CointegrationHit(
        y_symbol=y_symbol,
        x_symbol=x_symbol,
        pvalue=float(pvalue),
        test_statistic=float(score),
        critical_value_5pct=crit_5,
        hedge_ratio=beta,
        intercept=intercept,
        half_life=estimate_half_life(spread),
        n_obs=len(y),
        price_space=price_space,
    )


def engle_granger_pair(
    y_symbol: str,
    x_symbol: str,
    y_ohlcv: pd.DataFrame,
    x_ohlcv: pd.DataFrame,
    *,
    pvalue_threshold: float = 0.05,
    min_obs: int = 100,
    autolag: str | None = "aic",
    price_space: PriceSpace = "log",
) -> CointegrationHit | None:
    """Run Engle-Granger on (y, x). Return hit if p < threshold, else None."""
    hit = evaluate_pair(
        y_symbol,
        x_symbol,
        y_ohlcv,
        x_ohlcv,
        min_obs=min_obs,
        autolag=autolag,
        price_space=price_space,
    )
    if hit.pvalue >= pvalue_threshold:
        return None
    return hit
