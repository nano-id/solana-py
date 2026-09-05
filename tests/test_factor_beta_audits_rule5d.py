"""Tests for Module 3 post-Step-6 audits and Rule 5d helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pairs_trading.factor_beta.audits import (
    _decide_composite_vs_ada,
    audit_c1_selection_leakage,
    audit_c3_plateau_width,
)
from pairs_trading.factor_beta.config import load_factor_beta_config
from pairs_trading.factor_beta.rule5d import (
    null_residual_block_bootstrap,
    null_residual_rw,
)


def _pvalue_right(observed: float, null_stats: np.ndarray) -> dict:
    valid = null_stats[np.isfinite(null_stats)]
    n = int(len(valid))
    n_ge = int(np.sum(valid >= observed))
    return {"pvalue": float((1.0 + n_ge) / (n + 1.0))}


def test_decide_prefers_lower_std_when_both_pass_adf() -> None:
    rng = np.random.default_rng(0)
    n = 400
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + rng.normal(0, 1.0)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.5 * y[t - 1] + rng.normal(0, 0.6)
    out = _decide_composite_vs_ada(pd.Series(x), pd.Series(y), pmax=0.05)
    assert out["ada_passes_adf"] is True
    assert out["composite_passes_adf"] is True
    assert out["prefer_composite"] is True


def test_null_residual_preserves_length() -> None:
    idx = pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC")
    r = pd.Series(np.cumsum(np.random.default_rng(1).normal(0, 0.02, 100)), index=idx)
    rng = np.random.default_rng(2)
    rw = null_residual_rw(r, rng)
    bb = null_residual_block_bootstrap(r, rng)
    assert len(rw) == len(r)
    assert len(bb) == len(r)
    assert rw.index.equals(r.index)


def test_pvalue_right_tail_ranks_extreme() -> None:
    null = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert _pvalue_right(10.0, null)["pvalue"] == pytest.approx(1.0 / 6.0)
    assert _pvalue_right(-1.0, null)["pvalue"] == pytest.approx(1.0)


def test_audits_on_disk_artifacts() -> None:
    cfg = load_factor_beta_config("config/factor_beta.yaml")
    if not (Path(cfg.storage.output_dir) / "step6_walk_forward.json").is_file():
        pytest.skip("step6 artifact missing")
    c1 = audit_c1_selection_leakage(cfg)
    assert c1["interpretation_allowed"] is False
    assert c1["leakage_flags"]["original_decision_used_full_sample_including_oos"] is True
    c3 = audit_c3_plateau_width(cfg)
    assert c3["interpretation_allowed"] is False
    assert c3["selected"]["entry_z"] == 1.5
    assert "band_entry_pm_0_50_same_exit_z" in c3
