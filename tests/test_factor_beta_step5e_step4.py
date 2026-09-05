"""Tests for Module 3 Rule-5e clustering and Step-4 signal skeleton."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pairs_trading.data.storage import ParquetStore
from pairs_trading.factor_beta.clustering import _union_find_clusters, cluster_residuals
from pairs_trading.factor_beta.config import load_factor_beta_config
from pairs_trading.factor_beta.signals import (
    EntryGatePlaceholder,
    maybe_enter,
    run_cluster_signal_skeleton,
    run_synthetic_ou_mechanical_test,
)
from pairs_trading.signal.engine import SignalConfig


def test_union_find_clusters_merges_connected() -> None:
    nodes = ["A", "B", "C", "D"]
    edges = [("A", "B"), ("B", "C")]
    clusters = _union_find_clusters(nodes, edges)
    assert len(clusters) == 2
    big = next(c for c in clusters if len(c) == 3)
    assert set(big) == {"A", "B", "C"}
    assert ["D"] in clusters


def test_entry_gate_placeholder_blocks_when_edge_too_small() -> None:
    gate = EntryGatePlaceholder(
        min_edge_to_cost_multiple=2.5,
        round_trip_bps=35.0,
        expected_edge_bps=50.0,  # need 87.5
    )
    ok, reason = gate.allows_entry()
    assert ok is False
    assert "required" in reason

    gate_pass = EntryGatePlaceholder(
        min_edge_to_cost_multiple=2.5,
        round_trip_bps=35.0,
        expected_edge_bps=None,
    )
    ok2, reason2 = gate_pass.allows_entry()
    assert ok2 is True
    assert "placeholder" in reason2


def test_maybe_enter_requires_gate_in_signature() -> None:
    cfg = SignalConfig(entry_z=2.0, exit_z=0.5, stop_z=4.0, z_window=40)
    gate_block = EntryGatePlaceholder(2.5, 35.0, expected_edge_bps=10.0)
    allowed, reason, action = maybe_enter(
        z=-2.5, side_flat=True, signal_cfg=cfg, gate=gate_block
    )
    assert allowed is False
    assert action == "blocked"

    gate_ok = EntryGatePlaceholder(2.5, 35.0, expected_edge_bps=None)
    allowed2, _, action2 = maybe_enter(
        z=-2.5, side_flat=True, signal_cfg=cfg, gate=gate_ok
    )
    assert allowed2 is True
    assert action2 == "enter_long"


def test_synthetic_ou_mechanical() -> None:
    cfg = load_factor_beta_config("config/factor_beta.yaml")
    out = run_synthetic_ou_mechanical_test(cfg, n=3000, seed=7)
    assert out["passed"] is True
    assert out["n_enter"] > 0
    assert out["n_exit"] > 0
    assert out["rule5b_placeholder_invoked"] is True


def test_step5e_and_step4_on_disk() -> None:
    cfg = load_factor_beta_config("config/factor_beta.yaml")
    store = ParquetStore(Path(cfg.storage.root), cfg.storage.exchange_id, cfg.storage.timeframe)
    if not store.exists(cfg.factor.symbol):
        pytest.skip("OKX 1d data missing")
    cand = Path(cfg.storage.output_dir) / "step3_candidates.json"
    if not cand.is_file():
        pytest.skip("Step 3 candidates missing — run step3-adf first")

    cl = cluster_residuals(cfg)
    assert cl["interpretation_allowed"] is False
    assert cl["rules_still_locked"] is True
    assert cl["n_input_symbols"] == 7
    assert cl["n_independent_clusters"] >= 1
    assert cl["n_independent_clusters"] <= 7
    assert len(cl["signal_lines_recommended"]) == cl["n_independent_clusters"]
    # Rule 5e: every above-threshold pair must share a cluster
    thr = cl["threshold_abs_corr"]
    member_of = {}
    for c in cl["clusters"]:
        for m in c["members"]:
            member_of[m] = c["cluster_id"]
    for pair in cl["pairs_above_threshold"]:
        assert pair["abs_corr"] > thr
        assert member_of[pair["a"]] == member_of[pair["b"]]

    # Write lines file then run step 4
    from pairs_trading.factor_beta.clustering import save_cluster_report

    save_cluster_report(cl, Path(cfg.storage.output_dir))
    s4 = run_cluster_signal_skeleton(cfg)
    assert s4["step"] == 4
    assert s4["interpretation_allowed"] is False
    assert s4["n_signal_lines"] == cl["n_independent_clusters"]
    assert s4["rule5b_placeholder_present"] is True
    sk = s4["skeleton_thresholds_NOT_calibrated"]
    assert "walk-forward" in sk["source"].lower() or "Step 6" in sk["source"]
    # skeleton must come from config grids, not a hard-coded single number elsewhere
    assert sk["entry_z"] in cfg.signal.z_entry_grid
    assert sk["exit_z"] in cfg.signal.z_exit_grid
    assert sk["z_window"] in cfg.signal.zscore_window_grid
