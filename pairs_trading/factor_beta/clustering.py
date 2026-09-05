"""Rule 5e — residual correlation clustering on ADF survivors.

Coins with |corr(ε_i, ε_j)| > threshold are one signal, not independent.
Applied ONLY to Step-3 ADF pass list (not the full 20).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pairs_trading.factor_beta.config import FactorBetaConfig, load_factor_beta_config
from pairs_trading.factor_beta.factor_model import fit_rolling_factor


def _union_find_clusters(nodes: list[str], edges: list[tuple[str, str]]) -> list[list[str]]:
    parent = {n: n for n in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        union(a, b)
    groups: dict[str, list[str]] = {}
    for n in nodes:
        groups.setdefault(find(n), []).append(n)
    # stable order: sort members, sort clusters by first member
    return [sorted(v) for _, v in sorted(groups.items(), key=lambda kv: min(kv[1]))]


def load_adf_candidates(path: Path | None = None) -> dict:
    path = path or Path("data/okx/factor_beta/step3_candidates.json")
    return json.loads(path.read_text(encoding="utf-8"))


def cluster_residuals(
    config: FactorBetaConfig | None = None,
    *,
    symbols: list[str] | None = None,
    lookback_days: int | None = None,
    adf_pvalues: dict[str, float] | None = None,
) -> dict:
    """Pairwise |corr| on residuals; cluster edges above Rule-5e threshold."""
    config = config or load_factor_beta_config()
    lookback = int(lookback_days or config.factor.beta_lookback_days_default)
    thr = float(config.common_factor_control.max_abs_residual_corr)
    action = config.common_factor_control.action

    if symbols is None:
        cand = load_adf_candidates()
        symbols = list(cand["passed_symbols"])
    if not symbols:
        raise ValueError("No ADF-passed symbols to cluster")

    # ADF p-values for representative selection (lower = stronger stationarity reject)
    if adf_pvalues is None:
        adf_path = Path(config.storage.output_dir) / "step3_adf_stationarity_filter.json"
        if adf_path.is_file():
            adf = json.loads(adf_path.read_text(encoding="utf-8"))
            adf_pvalues = {
                r["symbol"]: float(r["adf"]["adf_pvalue"])
                for r in adf.get("per_coin", [])
                if r.get("adf", {}).get("adf_pvalue") is not None
            }
        else:
            adf_pvalues = {}

    result = fit_rolling_factor(config, lookback_days=lookback)
    missing = [s for s in symbols if s not in result.residual.columns]
    if missing:
        raise KeyError(f"Residuals missing for {missing}")

    resid = result.residual[symbols].dropna(how="any")
    corr = resid.corr()
    abs_corr = corr.abs()

    edges: list[tuple[str, str]] = []
    edge_rows: list[dict] = []
    for i, a in enumerate(symbols):
        for b in symbols[i + 1 :]:
            v = float(abs_corr.loc[a, b])
            edge_rows.append({"a": a, "b": b, "abs_corr": round(v, 4), "above": v > thr})
            if v > thr:
                edges.append((a, b))

    clusters_raw = _union_find_clusters(symbols, edges)
    clusters: list[dict] = []
    for idx, members in enumerate(clusters_raw, start=1):
        # representative = strongest ADF (lowest pvalue); fallback first alpha
        def _p(sym: str) -> float:
            return float(adf_pvalues.get(sym, 9.0))

        rep = min(members, key=_p)
        # also offer equal-weight cluster residual for optional use
        clusters.append(
            {
                "cluster_id": f"C{idx}",
                "members": members,
                "n_members": len(members),
                "representative": rep,
                "representative_rule": "lowest_adf_pvalue_among_members",
                "representative_adf_pvalue": adf_pvalues.get(rep),
                "is_singleton": len(members) == 1,
            }
        )

    # matrix for report
    matrix = {
        a: {b: round(float(abs_corr.loc[a, b]), 4) for b in symbols} for a in symbols
    }

    return {
        "step": "5e_early",
        "step_name": "residual_corr_clustering_on_adf_survivors",
        "status": "complete",
        "interpretation_allowed": False,
        "lookback_days": lookback,
        "threshold_abs_corr": thr,
        "action": action,
        "n_input_symbols": len(symbols),
        "input_symbols": symbols,
        "n_residual_bars": int(len(resid)),
        "n_pairs": len(edge_rows),
        "n_pairs_above_threshold": sum(1 for e in edge_rows if e["above"]),
        "pairs_above_threshold": [e for e in edge_rows if e["above"]],
        "pairs_all": edge_rows,
        "abs_corr_matrix": matrix,
        "n_independent_clusters": len(clusters),
        "clusters": clusters,
        "signal_lines_recommended": [
            {
                "cluster_id": c["cluster_id"],
                "signal_id": c["cluster_id"],
                "residual_source": "representative_member",
                "symbol": c["representative"],
                "members_covered": c["members"],
            }
            for c in clusters
        ],
        "rules_still_locked": True,
        "note": (
            f"Rule 5e on ADF survivors only: {len(symbols)} coins → "
            f"{len(clusters)} independent clusters. "
            "Step-4 z-skeleton should be one line per cluster, not per coin. "
            "Not a PnL / edge claim."
        ),
        "next_step": 4,
        "next_step_name": "cluster_zscore_signal_skeleton",
    }


def save_cluster_report(report: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "step5e_residual_clusters.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    # slim lines file for step 4
    lines = {
        "n_independent_clusters": report["n_independent_clusters"],
        "threshold_abs_corr": report["threshold_abs_corr"],
        "signal_lines": report["signal_lines_recommended"],
        "clusters": report["clusters"],
    }
    (output_dir / "step5e_signal_lines.json").write_text(
        json.dumps(lines, indent=2), encoding="utf-8"
    )
    return path
