"""Tests for Module 3 Steps 2b (β regime) and 3 (ADF filter)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pairs_trading.data.storage import ParquetStore
from pairs_trading.factor_beta.beta_regime import (
    _first_pos_to_neg_cross,
    analyze_beta_sign_flip,
)
from pairs_trading.factor_beta.config import load_factor_beta_config
from pairs_trading.factor_beta.stationarity import _adf_one, run_adf_filter


def test_first_pos_to_neg_cross_detects_flip() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    beta = pd.Series([1.0, 0.8, 0.2, -0.1, -0.5, 0.3, -0.2, -0.4, 0.1, 0.2], index=idx)
    cross = _first_pos_to_neg_cross(beta)
    assert cross is not None
    assert cross.date().isoformat() == "2024-01-04"


def test_adf_rejects_random_walk_accepts_ou_like() -> None:
    rng = np.random.default_rng(1)
    n = 400
    # Random walk → should fail stationarity at α=0.05 typically
    rw = pd.Series(np.cumsum(rng.normal(0, 1, n)))
    # Mean-reverting AR(1) with φ=0.5 → should pass
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + rng.normal(0, 1)
    ou = pd.Series(x)
    rw_adf = _adf_one(rw)
    ou_adf = _adf_one(ou)
    assert rw_adf["status"] == "ok"
    assert ou_adf["status"] == "ok"
    assert ou_adf["adf_pvalue"] < 0.05
    # RW p-value usually high; allow rare false reject but expect > 0.05 most seeds
    assert rw_adf["adf_pvalue"] > 0.01  # soft: not a hard fail if unlucky


def test_step2b_and_step3_on_disk() -> None:
    cfg = load_factor_beta_config("config/factor_beta.yaml")
    store = ParquetStore(Path(cfg.storage.root), cfg.storage.exchange_id, cfg.storage.timeframe)
    if not store.exists(cfg.factor.symbol):
        pytest.skip("OKX 1d data missing")
    diag = analyze_beta_sign_flip(cfg, symbols=["WLD/USDT", "NEAR/USDT"])
    assert diag["interpretation_allowed"] is False
    assert len(diag["per_symbol"]) == 2
    for row in diag["per_symbol"]:
        assert "classification" in row or row.get("status") == "missing_from_panel"

    adf = run_adf_filter(cfg, lookback_days=182)
    assert adf["step"] == 3
    assert adf["interpretation_allowed"] is False
    assert adf["rules_still_locked"] is True
    assert adf["n_coins_tested"] == 20
    assert adf["n_passed"] + adf["n_failed"] == 20
    assert adf["adf_pvalue_max"] == 0.05
