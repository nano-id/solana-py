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


def test_alt_index_and_btc_alt_residual():
    from pairs_trading.cointegration.neutralize import (
        build_btc_alt_residual_frames,
        build_equal_weight_alt_index,
    )

    rng = np.random.default_rng(3)
    n = 300
    log_btc = np.cumsum(rng.normal(0, 0.01, size=n)) + 10
    frames = {"BTC/USDT": _ohlcv(np.exp(log_btc)), "ETH/USDT": _ohlcv(np.exp(log_btc))}
    for i, sym in enumerate(["ADA/USDT", "SOL/USDT", "ARB/USDT", "DOGE/USDT"]):
        frames[sym] = _ohlcv(
            np.exp(log_btc + 0.3 * i + np.cumsum(rng.normal(0, 0.02, size=n)))
        )
    idx, members = build_equal_weight_alt_index(frames)
    assert "BTC/USDT" not in members and "ETH/USDT" not in members
    assert len(idx) == n - 1
    out, meta, _, _ = build_btc_alt_residual_frames(
        frames, min_obs=100, symbols=["ADA/USDT", "SOL/USDT"]
    )
    assert "ADA/USDT" in out
    assert "beta_btc" in meta["ADA/USDT"]
    assert "beta_alt_index" in meta["ADA/USDT"]


def test_by_on_p_bb_unique_dedupe():
    from pairs_trading.cointegration.residual_retest import apply_by_to_p_bb

    df = pd.DataFrame(
        {
            "y_symbol": ["BNB/USDT", "BNB/USDT", "ADA/USDT", "ADA/USDT"],
            "x_symbol": ["BTC/USDT", "BTC/USDT", "WLD/USDT", "ARB/USDT"],
            "pvalue_bb": [0.01, 0.03, 0.02, 0.04],
            "window_start": ["a", "b", "c", "d"],
            "window_end": ["a2", "b2", "c2", "d2"],
            "pair": ["BNB|BTC", "BNB|BTC", "ADA|WLD", "ADA|ARB"],
            "contains_bnb": [True, True, False, False],
        }
    )
    all_by, uniq, summary = apply_by_to_p_bb(df, alpha=0.05)
    assert summary["n_all"] == 4
    assert summary["n_unique"] == 3  # BNB-BTC deduped
    assert "significant_bb_by" in all_by.columns
    assert "significant_bb_by_unique" in uniq.columns
