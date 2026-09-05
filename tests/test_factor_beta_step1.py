"""Tests for Module 3 factor-beta Steps 1–2."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pairs_trading.data.storage import ParquetStore
from pairs_trading.factor_beta.config import CostsConfig, load_factor_beta_config
from pairs_trading.factor_beta.data import alignment_transparency, inventory_report
from pairs_trading.factor_beta.factor_model import (
    fit_rolling_factor,
    rolling_ols_alpha_beta_resid,
)


def test_rule5a_rejects_optimistic_5bps_cost() -> None:
    with pytest.raises(ValueError, match="Rule 5a"):
        CostsConfig(round_trip_bps_min=5.0, round_trip_bps_max=10.0, round_trip_bps=5.0)


def test_config_loads_with_rule5_and_timescale_separation() -> None:
    cfg = load_factor_beta_config("config/factor_beta.yaml")
    assert cfg.costs.round_trip_bps >= 30.0
    assert cfg.entry_gate.min_edge_to_cost_multiple >= 2.5
    assert cfg.evaluation.min_independent_windows >= 5
    assert cfg.multiple_testing.dual_null is True
    # Timescale separation: beta lookback ≫ trade weeks and ≠ z windows
    assert min(cfg.factor.beta_lookback_days_grid) >= 182
    assert cfg.evaluation.window_weeks_max <= 10
    assert max(cfg.signal.zscore_window_grid) < min(cfg.factor.beta_lookback_days_grid)


def test_alignment_is_inner_join_no_forward_fill() -> None:
    cfg = load_factor_beta_config("config/factor_beta.yaml")
    store = ParquetStore(Path(cfg.storage.root), cfg.storage.exchange_id, cfg.storage.timeframe)
    if not store.exists(cfg.factor.symbol):
        pytest.skip("OKX 1d data missing")
    align = alignment_transparency(cfg)
    assert align["forward_fill"] is False
    assert align["nan_fill"] is False
    assert "inner" in align["alignment_method"]
    assert align["aligned_bars"] == 999
    # Longer-history majors exist
    longer = set(align["symbols_with_longer_raw_history"])
    assert "BTC/USDT" in longer
    assert "ETH/USDT" in longer
    # Short-history binders exist
    binders = set(align["symbols_binding_aligned_start"])
    assert len(binders) >= 10


def test_rolling_ols_recovers_known_beta() -> None:
    rng = np.random.default_rng(0)
    n = 400
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    x = pd.Series(np.cumsum(rng.normal(0, 0.01, n)), index=idx)
    true_beta, true_alpha = 1.4, -0.2
    y = true_alpha + true_beta * x + pd.Series(rng.normal(0, 0.001, n), index=idx)
    a, b, e, r2 = rolling_ols_alpha_beta_resid(y, x, window=120, min_obs=120)
    # Last window should recover beta closely
    assert abs(float(b.dropna().iloc[-1]) - true_beta) < 0.05
    assert abs(float(a.dropna().iloc[-1]) - true_alpha) < 0.05
    assert float(r2.dropna().iloc[-1]) > 0.95


def test_step2_fit_on_disk() -> None:
    cfg = load_factor_beta_config("config/factor_beta.yaml")
    store = ParquetStore(Path(cfg.storage.root), cfg.storage.exchange_id, cfg.storage.timeframe)
    if not store.exists(cfg.factor.symbol):
        pytest.skip("OKX 1d data missing")
    inv = inventory_report(cfg)
    assert inv["interpretation_allowed"] is False
    assert inv["alignment"]["forward_fill"] is False
    result = fit_rolling_factor(cfg, lookback_days=182)
    assert result.lookback_days == 182
    assert result.beta.shape[1] >= 2
    # First lookback-1 rows should be NaN / invalid
    assert result.beta.isna().sum().min() >= 100
    assert result.residual.notna().sum().min() > 0
