"""Step 6 — per-line walk-forward z-threshold calibration.

Each Rule-5e signal line (C1/C2/C3) is calibrated independently.
Rule 5b is a hard gate: mean gross bps / trade ≥
min_edge_to_cost_multiple × round_trip_bps — failing cells are never
marked success (including heatmaps).

Plateau preference: among Rule-5b passers, prefer a peak whose 1-step
grid neighbors stay within plateau_neighbor_tolerance_pct of the peak
score (Module-2 lesson: distrust isolated spikes).
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pairs_trading.factor_beta.backtest import simulate_residual_pnl
from pairs_trading.factor_beta.c1_diagnostic import (
    equal_weight_composite,
    run_c1_diagnostic,
)
from pairs_trading.factor_beta.config import FactorBetaConfig, load_factor_beta_config
from pairs_trading.factor_beta.factor_model import fit_rolling_factor
from pairs_trading.signal.engine import SignalConfig


def _load_signal_lines(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_eval_windows(
    n_bars: int,
    *,
    window_days: int,
    min_windows: int,
    warmup: int,
) -> list[tuple[int, int]]:
    """Non-overlapping [start, end) windows after warmup."""
    start0 = int(warmup)
    usable = n_bars - start0
    if usable < window_days * min_windows:
        raise ValueError(
            f"Need ≥{min_windows}×{window_days} bars after warmup={warmup}; "
            f"have usable={usable}"
        )
    n_win = usable // window_days
    return [
        (start0 + k * window_days, start0 + (k + 1) * window_days)
        for k in range(n_win)
    ]


def _param_grid(config: FactorBetaConfig) -> list[SignalConfig]:
    stop = max(config.signal.z_entry_grid) + 1.0
    out: list[SignalConfig] = []
    for entry, exit_, zw in itertools.product(
        config.signal.z_entry_grid,
        config.signal.z_exit_grid,
        config.signal.zscore_window_grid,
    ):
        if exit_ >= entry:
            continue
        out.append(
            SignalConfig(
                entry_z=float(entry),
                exit_z=float(exit_),
                stop_z=float(stop),
                z_window=int(zw),
            )
        )
    return out


def _rule5b_min_edge_bps(config: FactorBetaConfig) -> float:
    return float(config.entry_gate.min_edge_to_cost_multiple) * float(
        config.costs.round_trip_bps
    )


def score_params_on_windows(
    residual: pd.Series,
    signal_cfg: SignalConfig,
    windows: list[tuple[int, int]],
    *,
    round_trip_bps: float,
    rule5b_min_edge_bps: float,
) -> dict[str, Any]:
    per_w: list[dict] = []
    all_trade_gross: list[float] = []
    for wi, (a, b) in enumerate(windows):
        bt = simulate_residual_pnl(
            residual,
            signal_cfg,
            round_trip_bps=round_trip_bps,
            start=a,
            end=b,
        )
        per_w.append(
            {
                "window": wi,
                "start": a,
                "end": b,
                "gross_bps": bt.gross_bps,
                "cost_bps": bt.cost_bps,
                "net_bps": bt.net_bps,
                "n_trades": bt.n_trades,
                "mean_gross_per_trade_bps": bt.mean_gross_per_trade_bps,
                "cost_leakage_ratio": (
                    bt.cost_leakage_ratio
                    if bt.cost_leakage_ratio == bt.cost_leakage_ratio
                    else None
                ),
            }
        )
        all_trade_gross.extend(t.gross_bps for t in bt.trades)

    n_trades = int(sum(w["n_trades"] for w in per_w))
    sum_gross = float(sum(w["gross_bps"] for w in per_w))
    sum_cost = float(sum(w["cost_bps"] for w in per_w))
    sum_net = float(sum(w["net_bps"] for w in per_w))
    mean_net = float(np.mean([w["net_bps"] for w in per_w])) if per_w else 0.0
    mean_gross = float(np.mean([w["gross_bps"] for w in per_w])) if per_w else 0.0
    expected_edge = float(np.mean(all_trade_gross)) if all_trade_gross else 0.0
    rule5b_ok = bool(n_trades > 0 and expected_edge >= rule5b_min_edge_bps)
    fail_reason = None
    if n_trades == 0:
        fail_reason = "no_trades"
    elif expected_edge < rule5b_min_edge_bps:
        fail_reason = (
            f"rule5b_fail expected_edge={expected_edge:.2f}bps "
            f"< {rule5b_min_edge_bps:.2f}bps"
        )

    return {
        "entry_z": signal_cfg.entry_z,
        "exit_z": signal_cfg.exit_z,
        "stop_z": signal_cfg.stop_z,
        "z_window": signal_cfg.z_window,
        "n_windows": len(windows),
        "n_trades": n_trades,
        "sum_gross_bps": round(sum_gross, 4),
        "sum_cost_bps": round(sum_cost, 4),
        "sum_net_bps": round(sum_net, 4),
        "mean_window_gross_bps": round(mean_gross, 4),
        "mean_window_net_bps": round(mean_net, 4),
        "expected_edge_bps_per_trade": round(expected_edge, 4),
        "rule5b_min_edge_bps": rule5b_min_edge_bps,
        "rule5b_pass": rule5b_ok,
        "success": rule5b_ok,
        "fail_reason": fail_reason,
        "per_window": per_w,
    }


def _neighbor_cells(cell: dict, grid_cells: list[dict]) -> list[dict]:
    entries = sorted({c["entry_z"] for c in grid_cells})
    exits = sorted({c["exit_z"] for c in grid_cells})
    zwins = sorted({c["z_window"] for c in grid_cells})

    def _adj(val: float, axis: list) -> list:
        if val not in axis:
            return []
        i = axis.index(val)
        out = []
        if i > 0:
            out.append(axis[i - 1])
        if i + 1 < len(axis):
            out.append(axis[i + 1])
        return out

    neigh: list[dict] = []
    for e in _adj(cell["entry_z"], entries):
        for c in grid_cells:
            if (
                c["entry_z"] == e
                and c["exit_z"] == cell["exit_z"]
                and c["z_window"] == cell["z_window"]
            ):
                neigh.append(c)
    for x in _adj(cell["exit_z"], exits):
        for c in grid_cells:
            if (
                c["entry_z"] == cell["entry_z"]
                and c["exit_z"] == x
                and c["z_window"] == cell["z_window"]
            ):
                neigh.append(c)
    for z in _adj(cell["z_window"], zwins):
        for c in grid_cells:
            if (
                c["entry_z"] == cell["entry_z"]
                and c["exit_z"] == cell["exit_z"]
                and c["z_window"] == z
            ):
                neigh.append(c)
    seen: set[tuple] = set()
    uniq: list[dict] = []
    for c in neigh:
        key = (c["entry_z"], c["exit_z"], c["z_window"])
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


def select_plateau(
    cells: list[dict],
    *,
    prefer_plateau: bool,
    tolerance_pct: float,
) -> dict[str, Any]:
    """Pick best Rule-5b-passing cell; flag isolated peaks."""
    passers = [c for c in cells if c.get("rule5b_pass")]
    if not passers:
        return {
            "selected": None,
            "selection_status": "no_rule5b_passers",
            "plateau_ok": False,
            "isolated_peak_suspicious": False,
            "note": "No parameter combo clears Rule 5b — nothing marked success.",
        }

    ranked = sorted(passers, key=lambda c: c["mean_window_net_bps"], reverse=True)
    peak = ranked[0]
    peak_score = float(peak["mean_window_net_bps"])
    neighbors = _neighbor_cells(peak, cells)
    pass_neighbors = [n for n in neighbors if n.get("rule5b_pass")]
    tol = max(abs(peak_score) * (tolerance_pct / 100.0), 1.0)
    close_neighbors = [
        n
        for n in pass_neighbors
        if abs(float(n["mean_window_net_bps"]) - peak_score) <= tol
    ]
    plateau_ok = len(close_neighbors) >= 1
    isolated = prefer_plateau and not plateau_ok

    selected = peak
    selection_note = "peak_rule5b_passer"
    if isolated:
        for cand in ranked:
            cand_score = float(cand["mean_window_net_bps"])
            cand_tol = max(abs(cand_score) * (tolerance_pct / 100.0), 1.0)
            nbs = [
                n
                for n in _neighbor_cells(cand, cells)
                if n.get("rule5b_pass")
                and abs(float(n["mean_window_net_bps"]) - cand_score) <= cand_tol
            ]
            if nbs:
                selected = cand
                selection_note = "plateau_preferred_over_isolated_peak"
                plateau_ok = True
                isolated = False
                break
        if isolated:
            selection_note = "isolated_peak_flagged_suspicious"

    return {
        "selected": {
            "entry_z": selected["entry_z"],
            "exit_z": selected["exit_z"],
            "stop_z": selected["stop_z"],
            "z_window": selected["z_window"],
            "mean_window_gross_bps": selected["mean_window_gross_bps"],
            "mean_window_net_bps": selected["mean_window_net_bps"],
            "sum_gross_bps": selected["sum_gross_bps"],
            "sum_cost_bps": selected["sum_cost_bps"],
            "sum_net_bps": selected["sum_net_bps"],
            "expected_edge_bps_per_trade": selected["expected_edge_bps_per_trade"],
            "n_trades": selected["n_trades"],
            "rule5b_pass": selected["rule5b_pass"],
        },
        "selection_status": selection_note,
        "plateau_ok": plateau_ok,
        "isolated_peak_suspicious": isolated,
        "n_rule5b_passers": len(passers),
        "peak_was": {
            "entry_z": peak["entry_z"],
            "exit_z": peak["exit_z"],
            "z_window": peak["z_window"],
            "mean_window_net_bps": peak["mean_window_net_bps"],
        },
        "close_neighbor_count": len(close_neighbors),
        "tolerance_pct": tolerance_pct,
    }


def expanding_oos_walk_forward(
    residual: pd.Series,
    grid: list[SignalConfig],
    windows: list[tuple[int, int]],
    *,
    config: FactorBetaConfig,
) -> dict[str, Any]:
    """Expanding train → select params → OOS test fold (per line)."""
    rt = float(config.costs.round_trip_bps)
    need = _rule5b_min_edge_bps(config)
    folds_out: list[dict] = []

    for test_i in range(1, len(windows)):
        train_wins = windows[:test_i]
        test_win = [windows[test_i]]
        train_scores = [
            score_params_on_windows(
                residual,
                cfg,
                train_wins,
                round_trip_bps=rt,
                rule5b_min_edge_bps=need,
            )
            for cfg in grid
        ]
        pick = select_plateau(
            train_scores,
            prefer_plateau=config.walk_forward.prefer_plateau,
            tolerance_pct=config.walk_forward.plateau_neighbor_tolerance_pct,
        )
        if pick["selected"] is None:
            folds_out.append(
                {
                    "test_fold": test_i,
                    "n_train_windows": len(train_wins),
                    "traded": False,
                    "skip_reason": "no_rule5b_passers_on_train",
                    "selected_params": None,
                    "oos": None,
                }
            )
            continue
        sel = pick["selected"]
        cfg = SignalConfig(
            entry_z=sel["entry_z"],
            exit_z=sel["exit_z"],
            stop_z=sel["stop_z"],
            z_window=int(sel["z_window"]),
        )
        oos = score_params_on_windows(
            residual,
            cfg,
            test_win,
            round_trip_bps=rt,
            rule5b_min_edge_bps=need,
        )
        folds_out.append(
            {
                "test_fold": test_i,
                "n_train_windows": len(train_wins),
                "traded": True,
                "skip_reason": None,
                "train_selection": {
                    "selection_status": pick["selection_status"],
                    "plateau_ok": pick["plateau_ok"],
                    "isolated_peak_suspicious": pick["isolated_peak_suspicious"],
                },
                "selected_params": sel,
                "oos": {
                    "gross_bps": oos["sum_gross_bps"],
                    "cost_bps": oos["sum_cost_bps"],
                    "net_bps": oos["sum_net_bps"],
                    "n_trades": oos["n_trades"],
                    "expected_edge_bps_per_trade": oos["expected_edge_bps_per_trade"],
                    "rule5b_pass": oos["rule5b_pass"],
                    "fail_reason": oos["fail_reason"],
                },
            }
        )

    traded = [f for f in folds_out if f.get("traded") and f.get("oos")]
    sum_gross = float(sum(f["oos"]["gross_bps"] for f in traded))
    sum_cost = float(sum(f["oos"]["cost_bps"] for f in traded))
    sum_net = float(sum(f["oos"]["net_bps"] for f in traded))
    return {
        "n_oos_folds": len(folds_out),
        "n_traded_folds": len(traded),
        "oos_sum_gross_bps": round(sum_gross, 4),
        "oos_sum_cost_bps": round(sum_cost, 4),
        "oos_sum_net_bps": round(sum_net, 4),
        "oos_mean_net_bps": round(sum_net / len(traded), 4) if traded else 0.0,
        "folds": folds_out,
    }


def resolve_line_residual(
    residuals: pd.DataFrame,
    line: dict,
    *,
    c1_source: str,
) -> tuple[pd.Series, dict]:
    cid = line["cluster_id"]
    members = list(line["members_covered"])
    if cid == "C1" and c1_source == "equal_weight_composite" and len(members) > 1:
        series = equal_weight_composite(residuals, members)
        meta = {
            "residual_source": "equal_weight_composite",
            "members": members,
            "label": "C1_composite",
        }
    else:
        sym = line["symbol"]
        series = residuals[sym].dropna()
        meta = {
            "residual_source": "representative_member",
            "symbol": sym,
            "members": members,
            "label": sym,
        }
    return series, meta


def run_walk_forward_all_lines(
    config: FactorBetaConfig | None = None,
    *,
    lookback_days: int | None = None,
    window_weeks: int | None = None,
) -> dict:
    config = config or load_factor_beta_config()
    lookback = int(lookback_days or config.factor.beta_lookback_days_default)
    if window_weeks is None:
        window_weeks = int(
            round(
                0.5
                * (
                    config.evaluation.window_weeks_min
                    + config.evaluation.window_weeks_max
                )
            )
        )
    window_days = int(window_weeks * 7)
    min_windows = int(config.evaluation.min_independent_windows)
    rt = float(config.costs.round_trip_bps)
    need = _rule5b_min_edge_bps(config)

    c1_diag = run_c1_diagnostic(config, lookback_days=lookback)
    c1_source = c1_diag["recommendation"]["c1_residual_source"]

    lines_path = Path(config.storage.output_dir) / "step5e_signal_lines.json"
    lines_doc = _load_signal_lines(lines_path)
    factor = fit_rolling_factor(config, lookback_days=lookback)
    grid = _param_grid(config)
    warmup = max(config.signal.zscore_window_grid)

    per_line: list[dict] = []
    for line in lines_doc["signal_lines"]:
        series, meta = resolve_line_residual(
            factor.residual, line, c1_source=c1_source
        )
        series = series.dropna()
        windows = build_eval_windows(
            len(series),
            window_days=window_days,
            min_windows=min_windows,
            warmup=warmup,
        )
        grid_scores = [
            score_params_on_windows(
                series,
                cfg,
                windows,
                round_trip_bps=rt,
                rule5b_min_edge_bps=need,
            )
            for cfg in grid
        ]
        heatmap = [
            {
                "entry_z": c["entry_z"],
                "exit_z": c["exit_z"],
                "z_window": c["z_window"],
                "mean_window_gross_bps": c["mean_window_gross_bps"],
                "mean_window_net_bps": c["mean_window_net_bps"],
                "sum_gross_bps": c["sum_gross_bps"],
                "sum_cost_bps": c["sum_cost_bps"],
                "sum_net_bps": c["sum_net_bps"],
                "expected_edge_bps_per_trade": c["expected_edge_bps_per_trade"],
                "n_trades": c["n_trades"],
                "rule5b_pass": c["rule5b_pass"],
                "success": c["success"],
                "fail_reason": c["fail_reason"],
            }
            for c in grid_scores
        ]
        in_sample_pick = select_plateau(
            grid_scores,
            prefer_plateau=config.walk_forward.prefer_plateau,
            tolerance_pct=config.walk_forward.plateau_neighbor_tolerance_pct,
        )
        oos = expanding_oos_walk_forward(series, grid, windows, config=config)

        n_pass = sum(1 for c in heatmap if c["rule5b_pass"])
        per_line.append(
            {
                "signal_id": line["signal_id"],
                "cluster_id": line["cluster_id"],
                "residual_meta": meta,
                "n_residual_bars": int(len(series)),
                "n_eval_windows": len(windows),
                "window_weeks": window_weeks,
                "window_days": window_days,
                "warmup_bars": warmup,
                "windows": [
                    {"i": i, "start": a, "end": b}
                    for i, (a, b) in enumerate(windows)
                ],
                "grid_n_combos": len(heatmap),
                "grid_n_rule5b_pass": n_pass,
                "grid_n_rule5b_fail": len(heatmap) - n_pass,
                "in_sample_all_windows_selection": in_sample_pick,
                "in_sample_note": (
                    "All-window grid is DIAGNOSTIC only — not OOS. "
                    "OOS claim lives in expanding_walk_forward."
                ),
                "expanding_walk_forward": oos,
                "heatmap_cells": heatmap,
            }
        )

    return {
        "step": 6,
        "step_name": "per_line_walk_forward_calibration",
        "status": "complete",
        "interpretation_allowed": False,
        "lookback_days": lookback,
        "costs_round_trip_bps": rt,
        "rule5b_min_edge_to_cost_multiple": (
            config.entry_gate.min_edge_to_cost_multiple
        ),
        "rule5b_min_expected_edge_bps": need,
        "evaluation": {
            "window_weeks": window_weeks,
            "window_days": window_days,
            "min_independent_windows": min_windows,
            "windows_non_overlapping": True,
        },
        "c1_diagnostic_summary": {
            "commonality_label": c1_diag["commonality_label"],
            "commonality_note": c1_diag["commonality_note"],
            "recommendation": c1_diag["recommendation"],
            "comparison": c1_diag["comparison"],
            "ada_adf_pvalue": c1_diag["ada_representative"]["adf"]["adf_pvalue"],
            "composite_adf_pvalue": c1_diag["equal_weight_composite"]["adf"][
                "adf_pvalue"
            ],
            "ada_resid_std": c1_diag["ada_representative"]["resid_std"],
            "composite_resid_std": c1_diag["equal_weight_composite"]["resid_std"],
        },
        "c1_residual_source_used": c1_source,
        "n_signal_lines": len(per_line),
        "per_line": per_line,
        "rules_still_locked": True,
        "rule5_reminder": {
            "5a": rt,
            "5b": need,
            "5c": min_windows,
            "5d": "dual_null+BY still pending (not this step)",
            "5e": "clusters fixed upstream",
        },
        "note": (
            "Per-line walk-forward only. Rule 5b failures are never success. "
            "Gross and net reported separately. No live capital. "
            "Dual-null FDR (5d) still outstanding before any claim."
        ),
        "next_step": "5d_or_review",
        "next_step_name": "dual_null_benjamini_yekutieli_or_human_review",
    }


def save_walk_forward_report(report: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "step6_walk_forward.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    slim_lines = []
    for line in report["per_line"]:
        slim_lines.append(
            {
                "signal_id": line["signal_id"],
                "residual_meta": line["residual_meta"],
                "grid_n_rule5b_pass": line["grid_n_rule5b_pass"],
                "grid_n_rule5b_fail": line["grid_n_rule5b_fail"],
                "in_sample_all_windows_selection": line[
                    "in_sample_all_windows_selection"
                ],
                "expanding_walk_forward": {
                    k: v
                    for k, v in line["expanding_walk_forward"].items()
                    if k != "folds"
                },
                "oos_folds_brief": [
                    {
                        "test_fold": f["test_fold"],
                        "traded": f["traded"],
                        "skip_reason": f.get("skip_reason"),
                        "params": f.get("selected_params"),
                        "oos": f.get("oos"),
                    }
                    for f in line["expanding_walk_forward"]["folds"]
                ],
            }
        )
    summary = {
        "step": 6,
        "interpretation_allowed": False,
        "c1_residual_source_used": report["c1_residual_source_used"],
        "c1_diagnostic_summary": report["c1_diagnostic_summary"],
        "rule5b_min_expected_edge_bps": report["rule5b_min_expected_edge_bps"],
        "costs_round_trip_bps": report["costs_round_trip_bps"],
        "per_line": slim_lines,
    }
    (output_dir / "step6_walk_forward_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return path
