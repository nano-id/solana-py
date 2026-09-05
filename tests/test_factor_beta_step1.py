"""Tests for Module 3 factor-beta Step 1 (data + Rule-5 lock-in)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pairs_trading.data.storage import ParquetStore
from pairs_trading.factor_beta.config import CostsConfig, load_factor_beta_config
from pairs_trading.factor_beta.data import build_log_price_panel, inventory_report


def test_rule5a_rejects_optimistic_5bps_cost() -> None:
    with pytest.raises(ValueError, match="Rule 5a"):
        CostsConfig(round_trip_bps_min=5.0, round_trip_bps_max=10.0, round_trip_bps=5.0)


def test_config_loads_with_rule5_locked() -> None:
    cfg = load_factor_beta_config("config/factor_beta.yaml")
    assert cfg.costs.round_trip_bps >= 30.0
    assert cfg.entry_gate.min_edge_to_cost_multiple >= 2.5
    assert cfg.evaluation.min_independent_windows >= 5
    assert cfg.evaluation.windows_must_be_non_overlapping is True
    assert cfg.multiple_testing.dual_null is True
    assert cfg.multiple_testing.fdr_method == "benjamini_yekutieli"
    assert cfg.common_factor_control.max_abs_residual_corr == 0.5


def test_step1_inventory_on_disk() -> None:
    cfg = load_factor_beta_config("config/factor_beta.yaml")
    store = ParquetStore(Path(cfg.storage.root), cfg.storage.exchange_id, cfg.storage.timeframe)
    if not store.exists(cfg.factor.symbol):
        pytest.skip("OKX 1d factor data not present")
    report = inventory_report(cfg)
    assert report["step"] == 1
    assert report["interpretation_allowed"] is False
    assert report["n_tradeable"] >= 2
    assert report["rules_locked"]["5a_costs_round_trip_bps"] >= 30.0
    panel = build_log_price_panel(cfg)
    assert cfg.factor.symbol in panel.log_prices.columns
    assert np.isfinite(panel.log_prices.to_numpy()).all()
