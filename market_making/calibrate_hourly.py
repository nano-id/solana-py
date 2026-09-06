"""Calibrate AS MM params on hourly (1h) candles.

Bar-denominated knobs (tau_bars, vol_lookback, ewma_halflife) are searched in
1h units. Prior 4h optima are not reused blindly.

Selection (IS months): maximize month×symbol positivity, then mean return,
then min return. OOS month(s) reported separately. Not a live-edge claim.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from market_making.config import MarketMakingConfig
from market_making.portfolio import run_portfolio_month
from market_making.simulator import run_month_sim


@dataclass
class HourlyCalibResult:
    selected_params: dict[str, float]
    is_months: list[tuple[int, int]]
    oos_months: list[tuple[int, int]]
    timeframe: str
    is_stats: dict[str, float]
    oos_stats: dict[str, float]
    portfolio_oos: dict[str, Any]
    top_candidates: list[dict[str, Any]]
    interpretation_allowed: bool
    method: str


def _apply_params(cfg: MarketMakingConfig, params: dict[str, float]) -> MarketMakingConfig:
    update = dict(params)
    if "vol_lookback" in update:
        update["vol_lookback"] = int(update["vol_lookback"])
    return cfg.model_copy(update=update)


def _eval_months(
    cfg: MarketMakingConfig,
    months: Iterable[tuple[int, int]],
    *,
    timeframe: str,
    seed: int,
) -> dict[str, float]:
    rets: list[float] = []
    halfs: list[float] = []
    for year, month in months:
        for sym in cfg.symbols:
            result, _ = run_month_sim(
                cfg,
                year=year,
                month=month,
                symbol=sym,
                timeframe=timeframe,
                seed=seed,
            )
            rets.append(float(result.total_return_pct))
            halfs.append(float(result.avg_half_spread_bps))
    arr = np.asarray(rets, dtype=float)
    return {
        "n": float(len(arr)),
        "pct_pos": float(100.0 * np.mean(arr > 0)) if len(arr) else 0.0,
        "mean_ret": float(arr.mean()) if len(arr) else 0.0,
        "min_ret": float(arr.min()) if len(arr) else 0.0,
        "mean_half": float(np.mean(halfs)) if halfs else 0.0,
    }


def default_1h_param_grid() -> list[dict[str, float]]:
    """Compact grid in 1h bar units (wall-clock aware vs prior 4h)."""
    gammas = [0.08, 0.12, 0.18]
    kappas = [0.8, 1.2]
    # Prior 4h tau=8 ≈ 32h wall → search nearby on 1h bars
    taus = [16.0, 32.0, 48.0]
    etas = [2.5, 3.5]
    advs = [0.35, 0.45]
    # Fixed structural 1h rescales for vol windows (~same wall as 4h 48/24)
    vol_lookback = 192
    ewma_halflife = 96.0
    out: list[dict[str, float]] = []
    for g, k, t, e, a in itertools.product(gammas, kappas, taus, etas, advs):
        out.append(
            {
                "gamma": float(g),
                "kappa": float(k),
                "tau_bars": float(t),
                "size_eta": float(e),
                "adverse_vol_mult": float(a),
                "vol_lookback": float(vol_lookback),
                "ewma_halflife": float(ewma_halflife),
                "liq_bps": 6.0,
                "vol_spread_mult": 0.25,
                "inventory_halt_frac": 0.80,
            }
        )
    return out


def _score(is_stats: dict[str, float]) -> tuple[float, float, float]:
    return (is_stats["pct_pos"], is_stats["mean_ret"], is_stats["min_ret"])


def calibrate_hourly(
    cfg: MarketMakingConfig,
    *,
    is_months: list[tuple[int, int]] | None = None,
    oos_months: list[tuple[int, int]] | None = None,
    timeframe: str = "1h",
    seed: int = 42,
    grid: list[dict[str, float]] | None = None,
    top_k: int = 8,
    portfolio_equity: float = 1000.0,
) -> HourlyCalibResult:
    is_months = is_months or [(2024, 12), (2025, 1), (2025, 2)]
    oos_months = oos_months or [(2025, 3)]
    grid = grid or default_1h_param_grid()

    ranked: list[tuple[tuple[float, float, float], dict[str, float], dict[str, float]]] = []
    for i, params in enumerate(grid):
        trial = _apply_params(cfg, params)
        is_stats = _eval_months(trial, is_months, timeframe=timeframe, seed=seed)
        ranked.append((_score(is_stats), params, is_stats))
        if (i + 1) % 12 == 0:
            print(f"calib progress {i + 1}/{len(grid)}", flush=True)

    ranked.sort(key=lambda x: x[0], reverse=True)
    best_params = ranked[0][1]
    best_is = ranked[0][2]
    best_cfg = _apply_params(cfg, best_params)
    oos_stats = _eval_months(best_cfg, oos_months, timeframe=timeframe, seed=seed)

    portfolio_oos: dict[str, Any] = {}
    for year, month in oos_months:
        pres, frame = run_portfolio_month(
            best_cfg,
            year=year,
            month=month,
            timeframe=timeframe,
            seed=seed,
            initial_equity=portfolio_equity,
        )
        holds = fifo_hold_hours_from_fills(frame, hours_per_bar=1.0 if timeframe == "1h" else 4.0)
        portfolio_oos[f"{year:04d}-{month:02d}"] = {
            "start_equity": pres.start_equity,
            "final_equity": pres.final_equity,
            "total_return_pct": pres.total_return_pct,
            "n_entries": pres.n_entries,
            "n_round_trips": pres.n_round_trips,
            "n_skipped_busy": pres.n_skipped_busy,
            "fees_paid": pres.fees_paid,
            "net_positive": pres.net_positive,
            "holds": holds,
        }

    top_candidates = [
        {"params": p, "is_stats": s, "score": list(sc)} for sc, p, s in ranked[:top_k]
    ]

    return HourlyCalibResult(
        selected_params={
            k: (int(v) if k == "vol_lookback" else float(v)) for k, v in best_params.items()
        },
        is_months=list(is_months),
        oos_months=list(oos_months),
        timeframe=timeframe,
        is_stats=best_is,
        oos_stats=oos_stats,
        portfolio_oos=portfolio_oos,
        top_candidates=top_candidates,
        interpretation_allowed=False,
        method=(
            f"1h candle AS grid; IS={is_months}; OOS={oos_months}; "
            "select by pct_pos→mean_ret→min_ret. Not a live-edge claim."
        ),
    )


def save_hourly_calib(
    result: HourlyCalibResult,
    *,
    root: str | Path = "data",
    exchange_id: str = "okx",
    basename: str = "hourly_calib_latest",
) -> dict[str, str]:
    out = Path(root) / exchange_id / "market_making"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{basename}.json"
    path.write_text(json.dumps(asdict(result), indent=2, default=str), encoding="utf-8")
    art = Path("/opt/cursor/artifacts")
    art.mkdir(parents=True, exist_ok=True)
    art_path = art / f"{basename}.json"
    art_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return {"summary": str(path), "artifact": str(art_path)}


def write_params_to_yaml(path: str | Path, params: dict[str, float]) -> None:
    """Merge selected params into market_making section of a YAML config."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    mm = dict(raw.get("market_making") or {})
    for k, v in params.items():
        mm[k] = int(v) if k == "vol_lookback" else float(v)
    raw["market_making"] = mm
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def fifo_hold_hours_from_fills(
    frame: pd.DataFrame,
    *,
    hours_per_bar: float = 1.0,
) -> dict[str, float | None]:
    """FIFO opposing-fill hold stats in hours from simulator/portfolio rows."""
    buy_col = "buy_fill" if "buy_fill" in frame.columns else "buy_fill"
    sell_col = "sell_fill" if "sell_fill" in frame.columns else "sell_fill"
    buy_sz = "buy_size" if "buy_size" in frame.columns else "buy_size"
    sell_sz = "sell_size" if "sell_size" in frame.columns else "sell_size"
    if frame.empty or buy_col not in frame.columns:
        return {"n_holds": 0, "mean_h": None, "median_h": None, "p25_h": None, "p75_h": None}

    buys: list[tuple[int, float]] = []
    holds: list[float] = []
    for i, row in frame.iterrows():
        bar = int(row["bar"]) if "bar" in frame.columns else int(i)
        if bool(row.get(buy_col)) and float(row.get(buy_sz, 0) or 0) > 0:
            buys.append((bar, float(row[buy_sz])))
        if bool(row.get(sell_col)) and float(row.get(sell_sz, 0) or 0) > 0:
            rem = float(row[sell_sz])
            while rem > 1e-12 and buys:
                b_bar, b_sz = buys[0]
                take = min(rem, b_sz)
                holds.append((bar - b_bar) * hours_per_bar)
                b_sz -= take
                rem -= take
                if b_sz <= 1e-12:
                    buys.pop(0)
                else:
                    buys[0] = (b_bar, b_sz)
    arr = np.asarray(holds, dtype=float)
    if len(arr) == 0:
        return {"n_holds": 0, "mean_h": None, "median_h": None, "p25_h": None, "p75_h": None}
    return {
        "n_holds": int(len(arr)),
        "mean_h": float(arr.mean()),
        "median_h": float(np.median(arr)),
        "p25_h": float(np.percentile(arr, 25)),
        "p75_h": float(np.percentile(arr, 75)),
    }
