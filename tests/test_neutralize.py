"""Tests for BTC factor-neutral residualization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pairs_trading.cointegration.neutralize import (
    build_factor_residual_frames,
    residualize_against_factor,
)


def _ohlcv(close: np.ndarray, start_ms: int = 1_700_000_000_000) -> pd.DataFrame:
    n = len(close)
    ts = np.arange(n) * 86_400_000 + start_ms
    return pd.DataFrame(
        {
            "timestamp": ts.astype(np.int64),
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.ones(n),
        }
    )


def test_residualize_recovers_orthogonal_noise():
    rng = np.random.default_rng(1)
    n = 400
    log_btc = np.cumsum(rng.normal(0, 0.01, size=n)) + 10
    eps = rng.normal(0, 0.005, size=n)
    log_coin = 1.0 + 0.5 * log_btc + eps
    resid_df, _alpha, beta, n_obs = residualize_against_factor(
        _ohlcv(np.exp(log_coin)),
        _ohlcv(np.exp(log_btc)),
        min_obs=100,
    )
    assert n_obs == n
    assert beta == pytest.approx(0.5, rel=0.05)
    corr = np.corrcoef(resid_df["close"].to_numpy(), log_btc)[0, 1]
    assert abs(corr) < 0.05


def test_build_factor_residual_frames_excludes_factor():
    rng = np.random.default_rng(2)
    n = 250
    log_btc = np.cumsum(rng.normal(0, 0.01, size=n)) + 10
    frames = {
        "BTC/USDT": _ohlcv(np.exp(log_btc)),
        "ETH/USDT": _ohlcv(np.exp(0.8 * log_btc + rng.normal(0, 0.01, size=n) + 1)),
        "SOL/USDT": _ohlcv(np.exp(1.2 * log_btc + rng.normal(0, 0.01, size=n) + 2)),
    }
    out, meta = build_factor_residual_frames(
        frames, factor_symbol="BTC/USDT", min_obs=100
    )
    assert "BTC/USDT" not in out
    assert set(out) == {"ETH/USDT", "SOL/USDT"}
    assert "beta_vs_factor" in meta["ETH/USDT"]
