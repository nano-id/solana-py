"""Rule 5d — dual-null + Benjamini–Yekutieli for Step-6 survivors.

Null A: independent Gaussian random-walk residual.
Null B: block-bootstrap of residual increments.

Statistic: expanding walk-forward OOS sum net bps (same protocol as Step 6).

C2/PEPE skipped (already OOS-negative). interpretation_allowed=false.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from pairs_trading.factor_beta.backtest import simulate_residual_pnl
from pairs_trading.factor_beta.c1_diagnostic import equal_weight_composite
from pairs_trading.factor_beta.config import FactorBetaConfig, load_factor_beta_config
from pairs_trading.factor_beta.factor_model import fit_rolling_factor
from pairs_trading.factor_beta.walkforward import (
    _param_grid,
    _rule5b_min_edge_bps,
    build_eval_windows,
    select_plateau,
)
from pairs_trading.signal.engine import SignalConfig


def _block_size(n: int) -> int:
    return int(max(5, min(n, round(np.sqrt(n)))))


def null_residual_rw(residual: pd.Series, rng: np.random.Generator) -> pd.Series:
    x = residual.to_numpy(dtype=float)
    diffs = np.diff(x)
    sigma = float(np.nanstd(diffs)) if len(diffs) else 1.0
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    shocks = rng.normal(0.0, sigma, size=len(x))
    path = np.cumsum(shocks)
    path = path - path[0] + float(x[0])
    return pd.Series(path, index=residual.index, name=residual.name)


def null_residual_block_bootstrap(
    residual: pd.Series, rng: np.random.Generator
) -> pd.Series:
    x = residual.to_numpy(dtype=float)
    diffs = np.diff(x)
    t = len(diffs)
    if t < 5:
        return null_residual_rw(residual, rng)
    bs = _block_size(t)
    n_blocks = int(np.ceil(t / bs))
    max_start = t - bs
    pieces = [
        diffs[int(rng.integers(0, max_start + 1)) : int(rng.integers(0, max_start + 1)) + bs]
        for _ in range(n_blocks)
    ]
    # fix: each piece needs its own start
    pieces = []
    for _ in range(n_blocks):
        start = int(rng.integers(0, max_start + 1))
        pieces.append(diffs[start : start + bs])
    boot = np.concatenate(pieces)[:t]
    path = np.concatenate([[float(x[0])], float(x[0]) + np.cumsum(boot)])
    return pd.Series(path, index=residual.index, name=residual.name)


def _matrices(
    residual: pd.Series,
    grid: list[SignalConfig],
    windows: list[tuple[int, int]],
    *,
    round_trip_bps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """gross, cost, net, n_trades, trade_gross_sum — shape (n_params, n_windows)."""
    n_p, n_w = len(grid), len(windows)
    gross = np.zeros((n_p, n_w))
    cost = np.zeros((n_p, n_w))
    net = np.zeros((n_p, n_w))
    ntr = np.zeros((n_p, n_w))
    tg = np.zeros((n_p, n_w))
    for pi, cfg in enumerate(grid):
        for wi, (a, b) in enumerate(windows):
            bt = simulate_residual_pnl(
                residual, cfg, round_trip_bps=round_trip_bps, start=a, end=b
            )
            gross[pi, wi] = bt.gross_bps
            cost[pi, wi] = bt.cost_bps
            net[pi, wi] = bt.net_bps
            ntr[pi, wi] = bt.n_trades
            tg[pi, wi] = float(sum(t.gross_bps for t in bt.trades))
    return gross, cost, net, ntr, tg


def _cells(
    grid: list[SignalConfig],
    idxs: list[int],
    *,
    gross: np.ndarray,
    net: np.ndarray,
    ntr: np.ndarray,
    tg: np.ndarray,
    cost: np.ndarray,
    rule5b_min: float,
) -> list[dict]:
    cells: list[dict] = []
    for pi, cfg in enumerate(grid):
        nt = int(ntr[pi, idxs].sum())
        expected = float(tg[pi, idxs].sum() / nt) if nt > 0 else 0.0
        ok = bool(nt > 0 and expected >= rule5b_min)
        cells.append(
            {
                "entry_z": cfg.entry_z,
                "exit_z": cfg.exit_z,
                "stop_z": cfg.stop_z,
                "z_window": cfg.z_window,
                "mean_window_gross_bps": float(gross[pi, idxs].mean()) if idxs else 0.0,
                "mean_window_net_bps": float(net[pi, idxs].mean()) if idxs else 0.0,
                "sum_gross_bps": float(gross[pi, idxs].sum()),
                "sum_cost_bps": float(cost[pi, idxs].sum()),
                "sum_net_bps": float(net[pi, idxs].sum()),
                "expected_edge_bps_per_trade": expected,
                "n_trades": nt,
                "rule5b_pass": ok,
            }
        )
    return cells


def expanding_oos_net(
    residual: pd.Series,
    config: FactorBetaConfig,
    *,
    windows: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    grid = _param_grid(config)
    if windows is None:
        ww = int(
            round(
                0.5
                * (
                    config.evaluation.window_weeks_min
                    + config.evaluation.window_weeks_max
                )
            )
        )
        windows = build_eval_windows(
            len(residual),
            window_days=int(ww * 7),
            min_windows=int(config.evaluation.min_independent_windows),
            warmup=max(config.signal.zscore_window_grid),
        )
    rt = float(config.costs.round_trip_bps)
    need = _rule5b_min_edge_bps(config)
    gross, cost, net, ntr, tg = _matrices(residual, grid, windows, round_trip_bps=rt)

    folds: list[dict] = []
    for test_i in range(1, len(windows)):
        train_cells = _cells(
            grid,
            list(range(test_i)),
            gross=gross,
            net=net,
            ntr=ntr,
            tg=tg,
            cost=cost,
            rule5b_min=need,
        )
        pick = select_plateau(
            train_cells,
            prefer_plateau=config.walk_forward.prefer_plateau,
            tolerance_pct=config.walk_forward.plateau_neighbor_tolerance_pct,
        )
        if pick.get("selected") is None:
            folds.append({"test_fold": test_i, "traded": False, "net_bps": 0.0, "gross_bps": 0.0, "cost_bps": 0.0})
            continue
        sel = pick["selected"]
        pi = next(
            i
            for i, cfg in enumerate(grid)
            if cfg.entry_z == sel["entry_z"]
            and cfg.exit_z == sel["exit_z"]
            and int(cfg.z_window) == int(sel["z_window"])
        )
        folds.append(
            {
                "test_fold": test_i,
                "traded": True,
                "net_bps": float(net[pi, test_i]),
                "gross_bps": float(gross[pi, test_i]),
                "cost_bps": float(cost[pi, test_i]),
                "selected": {
                    "entry_z": sel["entry_z"],
                    "exit_z": sel["exit_z"],
                    "z_window": sel["z_window"],
                },
            }
        )
    traded = [f for f in folds if f["traded"]]
    return {
        "oos_sum_gross_bps": float(sum(f["gross_bps"] for f in traded)),
        "oos_sum_cost_bps": float(sum(f["cost_bps"] for f in traded)),
        "oos_sum_net_bps": float(sum(f["net_bps"] for f in traded)),
        "n_traded_folds": len(traded),
        "folds": folds,
    }


def _pvalue_right(observed: float, null_stats: np.ndarray) -> dict[str, float | int]:
    valid = null_stats[np.isfinite(null_stats)]
    n = int(len(valid))
    if n == 0 or not np.isfinite(observed):
        return {"pvalue": 1.0, "n_valid": 0, "null_mean": float("nan")}
    n_ge = int(np.sum(valid >= observed))
    return {
        "pvalue": float((1.0 + n_ge) / (n + 1.0)),
        "n_valid": n,
        "null_mean": float(np.mean(valid)),
        "null_p95": float(np.quantile(valid, 0.95)),
        "null_p99": float(np.quantile(valid, 0.99)),
    }


def _worker(payload: dict) -> float:
    """Process worker: one null residual → OOS sum net."""
    config = load_factor_beta_config(payload["config_path"])
    residual = pd.Series(
        payload["values"],
        index=pd.to_datetime(payload["index"], utc=True),
        name=payload.get("name"),
    )
    rng = np.random.default_rng(int(payload["seed"]))
    if payload["kind"] == "rw":
        null_r = null_residual_rw(residual, rng)
    else:
        null_r = null_residual_block_bootstrap(residual, rng)
    return float(expanding_oos_net(null_r, config)["oos_sum_net_bps"])


def run_rule5d(
    config: FactorBetaConfig | None = None,
    *,
    n_sims: int = 80,
    seed: int = 42,
    signal_ids: list[str] | None = None,
    max_workers: int = 4,
    config_path: str | Path | None = None,
) -> dict:
    config = config or load_factor_beta_config(config_path)
    signal_ids = signal_ids or ["C1", "C3"]
    q = float(config.multiple_testing.fdr_q)
    lookback = int(config.factor.beta_lookback_days_default)
    cfg_path = str(Path(config_path or "config/factor_beta.yaml").resolve())

    lines_doc = json.loads(
        (Path(config.storage.output_dir) / "step5e_signal_lines.json").read_text()
    )
    factor = fit_rolling_factor(config, lookback_days=lookback)

    c1_source = "equal_weight_composite"
    summary = Path(config.storage.output_dir) / "step6_walk_forward_summary.json"
    if summary.is_file():
        c1_source = json.loads(summary.read_text()).get(
            "c1_residual_source_used", c1_source
        )

    per_line: list[dict] = []
    for line in lines_doc["signal_lines"]:
        sid = line["signal_id"]
        if sid not in signal_ids:
            continue
        members = list(line["members_covered"])
        if sid == "C1" and c1_source == "equal_weight_composite":
            series = equal_weight_composite(factor.residual, members).dropna()
            meta = {"residual_source": "equal_weight_composite", "members": members}
        else:
            series = factor.residual[line["symbol"]].dropna()
            meta = {
                "residual_source": "representative_member",
                "symbol": line["symbol"],
            }

        observed = expanding_oos_net(series, config)
        obs_net = float(observed["oos_sum_net_bps"])

        null_block: dict[str, Any] = {}
        for kind in ("rw", "block"):
            base = seed + (0 if kind == "rw" else 10_000)
            payloads = [
                {
                    "kind": kind,
                    "seed": base + i,
                    "values": series.to_numpy(dtype=float).tolist(),
                    "index": [str(ts) for ts in series.index],
                    "name": series.name,
                    "config_path": cfg_path,
                }
                for i in range(n_sims)
            ]
            stats = np.empty(n_sims, dtype=float)
            if max_workers <= 1:
                for i, p in enumerate(payloads):
                    stats[i] = _worker(p)
            else:
                with ProcessPoolExecutor(max_workers=max_workers) as pool:
                    futs = {pool.submit(_worker, p): i for i, p in enumerate(payloads)}
                    for fut in as_completed(futs):
                        stats[futs[fut]] = float(fut.result())
            pv = _pvalue_right(obs_net, stats)
            null_block[kind] = {
                "null_name": (
                    "independent_random_walk" if kind == "rw" else "block_bootstrap"
                ),
                "n_sims": n_sims,
                "observed_oos_sum_net_bps": obs_net,
                "null_mean_oos_sum_net_bps": pv["null_mean"],
                "null_p95_oos_sum_net_bps": pv.get("null_p95"),
                "null_p99_oos_sum_net_bps": pv.get("null_p99"),
                "pvalue": pv["pvalue"],
                "n_valid": pv["n_valid"],
            }

        p_dual = max(null_block["rw"]["pvalue"], null_block["block"]["pvalue"])
        per_line.append(
            {
                "signal_id": sid,
                "residual_meta": meta,
                "observed": {
                    "oos_sum_gross_bps": observed["oos_sum_gross_bps"],
                    "oos_sum_cost_bps": observed["oos_sum_cost_bps"],
                    "oos_sum_net_bps": observed["oos_sum_net_bps"],
                    "n_traded_folds": observed["n_traded_folds"],
                },
                "null_a_rw": null_block["rw"],
                "null_b_block_bootstrap": null_block["block"],
                "pvalue_dual_max": p_dual,
            }
        )

    pvals = [float(L["pvalue_dual_max"]) for L in per_line]
    if pvals:
        reject, p_adj, _, _ = multipletests(pvals, alpha=q, method="fdr_by")
    else:
        reject, p_adj = np.array([], dtype=bool), np.array([])

    for i, L in enumerate(per_line):
        L["pvalue_dual_by"] = float(p_adj[i])
        L["significant_by_at_q"] = bool(reject[i])
        L["survives_rule5d"] = bool(reject[i])

    n_survive = int(sum(1 for L in per_line if L["survives_rule5d"]))
    return {
        "step": "5d_dual_null_benjamini_yekutieli",
        "status": "complete",
        "interpretation_allowed": False,
        "n_sims_per_null": n_sims,
        "seed": seed,
        "fdr_method": "benjamini_yekutieli",
        "fdr_q": q,
        "null_a": "independent_random_walk",
        "null_b": "block_bootstrap",
        "lines_tested": signal_ids,
        "lines_skipped": ["C2"],
        "skip_reason_c2": "OOS net already negative in Step 6",
        "c1_residual_source_used": c1_source,
        "n_lines_tested": len(per_line),
        "n_lines_survive_by": n_survive,
        "per_line": per_line,
        "verdict": (
            "NO_LINE_SURVIVES_RULE5D"
            if n_survive == 0
            else "SOME_LINES_SURVIVE_RULE5D_STILL_NO_CLAIM"
        ),
        "note": (
            "Rule 5d control only. Dual p = max(p_rw, p_block); BY across tested "
            "lines. Raw OOS bps magnitude is not a claim. interpretation_allowed=false."
        ),
    }


def save_rule5d_report(report: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "step5d_dual_null_by.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
