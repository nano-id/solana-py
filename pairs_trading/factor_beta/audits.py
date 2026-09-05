"""Post-Step-6 audits: C1 selection leakage + C3 plateau width.

interpretation_allowed remains false — controls only, not green lights.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # noqa: F401 — index typing / Timestamp helpers

from pairs_trading.factor_beta.c1_diagnostic import (
    C1_MEMBER_PROFILES,
    equal_weight_composite,
)
from pairs_trading.factor_beta.config import FactorBetaConfig, load_factor_beta_config
from pairs_trading.factor_beta.factor_model import fit_rolling_factor
from pairs_trading.factor_beta.stationarity import _adf_one
from pairs_trading.factor_beta.walkforward import build_eval_windows


def _decide_composite_vs_ada(
    ada: pd.Series,
    composite: pd.Series,
    *,
    pmax: float,
) -> dict[str, Any]:
    """Same decision rule as run_c1_diagnostic (keep in sync)."""
    adf_ada = _adf_one(ada)
    adf_comp = _adf_one(composite)
    ada_pass = bool(
        adf_ada["status"] == "ok"
        and adf_ada["adf_pvalue"] is not None
        and adf_ada["adf_pvalue"] <= pmax
    )
    comp_pass = bool(
        adf_comp["status"] == "ok"
        and adf_comp["adf_pvalue"] is not None
        and adf_comp["adf_pvalue"] <= pmax
    )
    std_ada = float(ada.std(ddof=0))
    std_comp = float(composite.std(ddof=0))
    p_ada = float(adf_ada["adf_pvalue"]) if adf_ada["adf_pvalue"] is not None else 9.0
    p_comp = float(adf_comp["adf_pvalue"]) if adf_comp["adf_pvalue"] is not None else 9.0

    prefer_composite = False
    reasons: list[str] = []
    if comp_pass and (not ada_pass):
        prefer_composite = True
        reasons.append("composite_passes_ADF_ADA_fails")
    elif comp_pass and ada_pass:
        if p_comp <= p_ada:
            prefer_composite = True
            reasons.append("composite_adf_pvalue_leq_ADA")
        if std_comp < std_ada * 0.95:
            prefer_composite = True
            reasons.append("composite_std_lower")
        if not prefer_composite:
            if p_comp <= max(p_ada * 2.0, p_ada + 0.01):
                prefer_composite = True
                reasons.append("both_pass_composite_comparable_noise_avg_preferred")
    elif ada_pass and not comp_pass:
        reasons.append("keep_ADA_composite_fails_ADF")
    else:
        reasons.append("neither_clear_keep_ADA_rep")
    if not reasons:
        reasons.append("default_ADA_rep")

    return {
        "n_bars": int(len(ada)),
        "start": str(pd.Timestamp(ada.index.min()).date()) if len(ada) else None,
        "end": str(pd.Timestamp(ada.index.max()).date()) if len(ada) else None,
        "adf_pvalue_ada": p_ada,
        "adf_pvalue_composite": p_comp,
        "resid_std_ada": std_ada,
        "resid_std_composite": std_comp,
        "std_ratio_composite_vs_ada": round(std_comp / std_ada, 4) if std_ada > 0 else None,
        "ada_passes_adf": ada_pass,
        "composite_passes_adf": comp_pass,
        "prefer_composite": prefer_composite,
        "c1_residual_source": (
            "equal_weight_composite" if prefer_composite else "representative_ADA"
        ),
        "reasons": reasons,
    }


def audit_c1_selection_leakage(
    config: FactorBetaConfig | None = None,
    *,
    lookback_days: int | None = None,
) -> dict:
    """Full-sample vs first-IS-only composite/ADA decision (look-ahead check)."""
    config = config or load_factor_beta_config()
    lookback = int(lookback_days or config.factor.beta_lookback_days_default)
    members = list(C1_MEMBER_PROFILES.keys())
    pmax = float(config.stationarity.adf_pvalue_max)

    result = fit_rolling_factor(config, lookback_days=lookback)
    resid = result.residual[members].dropna(how="any")
    ada_full = resid["ADA/USDT"]
    comp_full = equal_weight_composite(result.residual, members)

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
    warmup = max(config.signal.zscore_window_grid)
    windows = build_eval_windows(
        len(resid),
        window_days=window_days,
        min_windows=int(config.evaluation.min_independent_windows),
        warmup=warmup,
    )
    a0, b0 = windows[0]

    full = _decide_composite_vs_ada(ada_full, comp_full, pmax=pmax)
    first_window_only = _decide_composite_vs_ada(
        ada_full.iloc[a0:b0], comp_full.iloc[a0:b0], pmax=pmax
    )
    through_first_is = _decide_composite_vs_ada(
        ada_full.iloc[:b0], comp_full.iloc[:b0], pmax=pmax
    )

    step6_used = "equal_weight_composite"
    artifact = Path(config.storage.output_dir) / "step6_walk_forward_summary.json"
    if artifact.is_file():
        prev = json.loads(artifact.read_text(encoding="utf-8"))
        step6_used = prev.get(
            "c1_residual_source_used",
            prev.get("c1_residual_source_used", step6_used),
        )

    leak_vs_strict = step6_used != first_window_only["c1_residual_source"]
    leak_vs_through = step6_used != through_first_is["c1_residual_source"]
    full_matches_step6 = step6_used == full["c1_residual_source"]

    if leak_vs_strict or leak_vs_through:
        verdict = "LEAKAGE_CONFIRMED_SOURCE_CHANGES_UNDER_IS_ONLY"
    elif full_matches_step6:
        verdict = "LEAKAGE_IN_PROCEDURE_SOURCE_UNCHANGED_UNDER_IS_ONLY"
    else:
        verdict = "REVIEW"

    return {
        "step": "c1_selection_leakage_audit",
        "status": "complete",
        "interpretation_allowed": False,
        "lookback_days": lookback,
        "members": members,
        "residual_span_full": {
            "n_bars": int(len(resid)),
            "start": str(resid.index.min().date()),
            "end": str(resid.index.max().date()),
        },
        "first_wf_train_window": {
            "window_index": 0,
            "bar_start": a0,
            "bar_end": b0,
            "start": str(resid.index[a0].date()),
            "end": str(resid.index[b0 - 1].date()),
            "n_bars": int(b0 - a0),
        },
        "decision_full_sample": full,
        "decision_first_window_only_strict": first_window_only,
        "decision_through_end_of_first_is": through_first_is,
        "step6_c1_source_used": step6_used,
        "full_sample_matches_step6_source": full_matches_step6,
        "leakage_flags": {
            "original_decision_used_full_sample_including_oos": True,
            "source_changes_if_restricted_to_first_window_only": leak_vs_strict,
            "source_changes_if_restricted_through_first_is_end": leak_vs_through,
        },
        "verdict": verdict,
        "note": (
            "Control only. Original C1 diagnostic spanned the full residual "
            "history (overlaps all OOS folds) — procedural look-ahead. "
            "IS-only recompute shown for whether the *choice* flips. No edge claim."
        ),
    }


def audit_c3_plateau_width(
    config: FactorBetaConfig | None = None,
    *,
    step6_path: Path | None = None,
) -> dict:
    """Wider-band plateau check around C3 selected cell (entry_z=1.5)."""
    config = config or load_factor_beta_config()
    path = step6_path or Path(config.storage.output_dir) / "step6_walk_forward.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    c3 = next(L for L in doc["per_line"] if L["signal_id"] == "C3")
    selected = c3["in_sample_all_windows_selection"]["selected"]
    cells = c3["heatmap_cells"]
    e0 = float(selected["entry_z"])
    x0 = float(selected["exit_z"])
    z0 = int(selected["z_window"])
    peak_net = float(selected["mean_window_net_bps"])
    tol_pct = float(config.walk_forward.plateau_neighbor_tolerance_pct)
    tol = max(abs(peak_net) * (tol_pct / 100.0), 1.0)

    def _summarize(band_cells: list[dict], *, same_exit_and_z: bool) -> dict:
        rows = []
        for c in band_cells:
            if same_exit_and_z and (
                float(c["exit_z"]) != x0 or int(c["z_window"]) != z0
            ):
                continue
            net = float(c["mean_window_net_bps"])
            rows.append(
                {
                    "entry_z": c["entry_z"],
                    "exit_z": c["exit_z"],
                    "z_window": c["z_window"],
                    "mean_window_gross_bps": c["mean_window_gross_bps"],
                    "mean_window_net_bps": c["mean_window_net_bps"],
                    "expected_edge_bps_per_trade": c["expected_edge_bps_per_trade"],
                    "n_trades": c["n_trades"],
                    "rule5b_pass": c["rule5b_pass"],
                    "success": c["success"],
                    "within_tol_of_peak": abs(net - peak_net) <= tol,
                    "net_vs_peak_bps": round(net - peak_net, 4),
                }
            )
        rows = sorted(rows, key=lambda r: (r["entry_z"], r["exit_z"], r["z_window"]))
        nets = [r["mean_window_net_bps"] for r in rows if r["rule5b_pass"]]
        return {
            "n_cells": len(rows),
            "n_rule5b_pass": sum(1 for r in rows if r["rule5b_pass"]),
            "n_within_tol_and_pass": sum(
                1 for r in rows if r["within_tol_of_peak"] and r["rule5b_pass"]
            ),
            "net_min": min(nets) if nets else None,
            "net_max": max(nets) if nets else None,
            "net_median": float(np.median(nets)) if nets else None,
            "peak_net": peak_net,
            "tolerance_bps": round(tol, 4),
            "cells": rows,
        }

    band_cells_pm50 = [
        c for c in cells if abs(float(c["entry_z"]) - e0) <= 0.5 + 1e-12
    ]
    band_pm25_same = _summarize(
        [c for c in cells if abs(float(c["entry_z"]) - e0) <= 0.25 + 1e-12],
        same_exit_and_z=True,
    )
    band_pm50_same = _summarize(band_cells_pm50, same_exit_and_z=True)
    band_pm50_any = _summarize(band_cells_pm50, same_exit_and_z=False)
    entry15 = _summarize(
        [c for c in cells if abs(float(c["entry_z"]) - e0) < 1e-12],
        same_exit_and_z=False,
    )

    # Neighbor at entry=2.0 same exit/z vs peak at 1.5
    neighbor_drop = None
    for r in band_pm50_same["cells"]:
        if float(r["entry_z"]) == 2.0:
            neighbor_drop = r["net_vs_peak_bps"]

    weak = band_pm50_same["n_within_tol_and_pass"] <= 1 or (
        neighbor_drop is not None and abs(neighbor_drop) > tol
    )

    return {
        "step": "c3_plateau_width_audit",
        "status": "complete",
        "interpretation_allowed": False,
        "selected": selected,
        "selection_status": c3["in_sample_all_windows_selection"]["selection_status"],
        "close_neighbor_count_from_step6": c3["in_sample_all_windows_selection"].get(
            "close_neighbor_count"
        ),
        "grid_entry_spacing": 0.5,
        "note_on_pm25": (
            "entry grid is {1.5,2.0,2.5,3.0}; ±0.25 has no adjacent grid point. "
            "±0.5 is the first real neighbor band (entry 2.0)."
        ),
        "band_entry_pm_0_25_same_exit_z": band_pm25_same,
        "band_entry_pm_0_50_same_exit_z": band_pm50_same,
        "band_entry_pm_0_50_any_exit_z": band_pm50_any,
        "all_cells_at_entry_1_5": entry15,
        "neighbor_entry_2_0_net_vs_peak_bps": neighbor_drop,
        "plateau_strength": "weak" if weak else "moderate",
        "verdict": (
            "WEAK_PLATEAU_NARROW_SUPPORT" if weak else "BROADER_BAND_SUPPORTIVE"
        ),
        "note": (
            "Control only. entry_z=1.5 is the most aggressive grid edge. "
            "Wide-band consistency required before treating C3 as a plateau. "
            "No edge claim."
        ),
    }


def save_audit_report(report: dict, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
