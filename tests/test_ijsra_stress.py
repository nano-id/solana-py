"""Unit tests for IJSRA 2026 × Rule 5d stress helpers (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pairs_trading.cointegration.ijsra_stress import (
    _annualize,
    _orient,
    _pair_label,
    _right_tail_pvalue,
    aligned_closes,
    dual_null_strategy_ann_net,
    eg_paper_pair,
    paper_pairs_backtest,
)


def _synthetic_frames(n: int = 400, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Independent RW prices — should NOT look cointegrated."""
    rng = np.random.default_rng(seed)
    ts = np.arange(n, dtype=np.int64) * 86_400_000 + 1_640_995_200_000  # ~2022-01-01
    frames: dict[str, pd.DataFrame] = {}
    for sym, level in [
        ("BTC/USDT", 40_000.0),
        ("ETH/USDT", 3_000.0),
        ("LTC/USDT", 100.0),
        ("XRP/USDT", 0.5),
    ]:
        shocks = rng.normal(0.0, 0.02, size=n)
        logp = np.log(level) + np.cumsum(shocks)
        frames[sym] = pd.DataFrame(
            {
                "timestamp": ts,
                "close": np.exp(logp),
            }
        )
    return frames


def test_pair_label_and_orient() -> None:
    assert _pair_label("ETH/USDT", "BTC/USDT") == "BTC-ETH"
    y, x = _orient("BTC/USDT", "ETH/USDT")
    assert y == "ETH/USDT" and x == "BTC/USDT"


def test_annualize_and_right_tail() -> None:
    assert abs(_annualize(0.0, 365) - 0.0) < 1e-12
    assert _right_tail_pvalue(1.0, np.array([0.0, 0.5, 0.9])) < 0.5
    assert _right_tail_pvalue(10.0, np.array([0.0, 0.5, 0.9])) < 0.3


def test_eg_independent_rw_not_significant() -> None:
    frames = _synthetic_frames()
    row = eg_paper_pair(frames, "ETH/USDT", "BTC/USDT", price_space="log")
    assert row["pair_label"] == "BTC-ETH"
    assert row["n_obs"] == 400
    # Independent walks: table p should not be tiny
    assert row["pvalue_table"] > 0.01


def test_paper_backtest_runs_and_charges_costs() -> None:
    frames = _synthetic_frames(n=500, seed=1)
    bt = paper_pairs_backtest(frames, "ETH/USDT", "BTC/USDT", price_space="log")
    assert bt.pair_label == "BTC-ETH"
    assert bt.n_obs == 500
    assert bt.total_cost >= 0.0
    assert np.isfinite(bt.ann_net)


def test_dual_null_strategy_returns_pvalues() -> None:
    frames = _synthetic_frames(n=300, seed=2)
    m = aligned_closes(frames, "ETH/USDT", "BTC/USDT")
    log_y = np.log(m["y"].to_numpy(dtype=float))
    log_x = np.log(m["x"].to_numpy(dtype=float))
    bt = paper_pairs_backtest(frames, "ETH/USDT", "BTC/USDT")
    null = dual_null_strategy_ann_net(
        log_y,
        log_x,
        observed_ann_net=bt.ann_net,
        n_sims=20,
        seed=7,
    )
    assert 0.0 < null["pvalue_dual"] <= 1.0
    assert null["n_sims"] == 20
    assert null["n_valid_rw"] > 0
