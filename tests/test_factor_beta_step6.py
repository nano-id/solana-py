"""Tests for Module 3 C1 diagnostic and Step-6 walk-forward."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pairs_trading.factor_beta.backtest import simulate_residual_pnl
from pairs_trading.factor_beta.c1_diagnostic import (
    equal_weight_composite,
    run_c1_diagnostic,
)
from pairs_trading.factor_beta.config import load_factor_beta_config
from pairs_trading.factor_beta.walkforward import (
    _rule5b_min_edge_bps,
    build_eval_windows,
    resolve_line_residual,
    score_params_on_windows,
    select_plateau,
)
from pairs_trading.signal.engine import SignalConfig
from pairs_trading.signal.ou import simulate_ou_spread


def test_equal_weight_composite_averages() -> None:
    import pandas as pd

    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    df = pd.DataFrame(
        {"A/USDT": [1.0, 2.0, 3.0, 4.0, 5.0], "B/USDT": [3.0, 4.0, 5.0, 6.0, 7.0]},
        index=idx,
    )
    comp = equal_weight_composite(df, ["A/USDT", "B/USDT"])
    assert list(comp.to_numpy()) == [2.0, 3.0, 4.0, 5.0, 6.0]


def test_rule5b_threshold_is_multiple_of_rt() -> None:
    cfg = load_factor_beta_config("config/factor_beta.yaml")
    need = _rule5b_min_edge_bps(cfg)
    assert need == pytest.approx(
        cfg.entry_gate.min_edge_to_cost_multiple * cfg.costs.round_trip_bps
    )
    assert need >= 87.5  # 2.5 × 35


def test_build_eval_windows_non_overlapping() -> None:
    wins = build_eval_windows(500, window_days=63, min_windows=5, warmup=60)
    assert len(wins) >= 5
    for (a, b), (c, d) in zip(wins, wins[1:]):
        assert b == c
        assert b - a == 63


def test_select_plateau_rejects_empty_passers() -> None:
    cells = [
        {
            "entry_z": 2.0,
            "exit_z": 0.5,
            "stop_z": 4.0,
            "z_window": 40,
            "mean_window_net_bps": 10.0,
            "mean_window_gross_bps": 50.0,
            "sum_gross_bps": 50.0,
            "sum_cost_bps": 35.0,
            "sum_net_bps": 15.0,
            "expected_edge_bps_per_trade": 50.0,
            "n_trades": 1,
            "rule5b_pass": False,
        }
    ]
    pick = select_plateau(cells, prefer_plateau=True, tolerance_pct=10.0)
    assert pick["selected"] is None
    assert pick["selection_status"] == "no_rule5b_passers"


def test_ou_residual_pnl_charges_round_trip() -> None:
    spread = simulate_ou_spread(n=2000, theta=0.15, mu=0.0, sigma=0.04, seed=3)
    cfg = SignalConfig(entry_z=2.0, exit_z=0.5, stop_z=4.0, z_window=40)
    bt = simulate_residual_pnl(spread, cfg, round_trip_bps=35.0)
    assert bt.n_trades > 0
    assert bt.cost_bps == pytest.approx(35.0 * bt.n_trades)
    assert bt.net_bps == pytest.approx(bt.gross_bps - bt.cost_bps)


def test_c1_diagnostic_and_rule5b_scoring_on_disk() -> None:
    cfg = load_factor_beta_config("config/factor_beta.yaml")
    out = Path(cfg.storage.output_dir)
    lines = out / "step5e_signal_lines.json"
    if not lines.is_file():
        pytest.skip("step5e artifacts missing")

    c1 = run_c1_diagnostic(cfg)
    assert c1["interpretation_allowed"] is False
    assert "partial" in c1["commonality_label"] or c1["commonality_label"] in {
        "homogeneous_sector",
        "partial_overlap_not_homogeneous",
        "heterogeneous_no_clear_sector",
    }
    assert "c1_residual_source" in c1["recommendation"]
    assert c1["ada_representative"]["passes_adf"] is True
    assert c1["equal_weight_composite"]["passes_adf"] is True

    from pairs_trading.factor_beta.factor_model import fit_rolling_factor

    factor = fit_rolling_factor(cfg)
    lines_doc = json.loads(lines.read_text())
    line0 = lines_doc["signal_lines"][0]
    series, _meta = resolve_line_residual(
        factor.residual,
        line0,
        c1_source=c1["recommendation"]["c1_residual_source"],
    )
    series = series.dropna()
    wins = build_eval_windows(
        len(series), window_days=63, min_windows=5, warmup=60
    )
    need = _rule5b_min_edge_bps(cfg)
    scored = score_params_on_windows(
        series,
        SignalConfig(entry_z=2.5, exit_z=0.5, stop_z=4.0, z_window=40),
        wins,
        round_trip_bps=cfg.costs.round_trip_bps,
        rule5b_min_edge_bps=need,
    )
    assert scored["success"] == scored["rule5b_pass"]
    assert "sum_gross_bps" in scored and "sum_net_bps" in scored
    if scored["expected_edge_bps_per_trade"] < need:
        assert scored["rule5b_pass"] is False
