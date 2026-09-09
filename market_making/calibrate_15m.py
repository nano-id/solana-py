"""Calibrate AS MM params on 15-minute candles.

Bar knobs (tau_bars, vol_lookback, ewma_halflife) are searched in 15m units.
Wall-clock targets match the frozen 1h package (×4 bars).

Selection (IS months): maximize month×symbol positivity, then mean return,
then min return. OOS month(s) reported separately. Not a live-edge claim.

The frozen 1h lock (config/LOCKED_HOURLY_V2.md) is not modified by this module.
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

HOURS_PER_BAR = 0.25
BARS_PER_YEAR = 35_040.0  # 365.25 * 24 * 4


@dataclass
class FifteenMinCalibResult:
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
    hours_per_bar: float = HOURS_PER_BAR
    bars_per_year: float = BARS_PER_YEAR


def _apply_params(cfg: MarketMakingConfig, params: dict[str, float]) -> MarketMakingConfig:
    update = dict(params)
    if "vol_lookback" in update:
        update["vol_lookback"] = int(update["vol_lookback"])
    # Keep sim fill model aligned with locked 1h retail book; fix bars/year for 15m.
    sim = cfg.sim.model_copy(
        update={
            "bars_per_year": BARS_PER_YEAR,
            "fill_prob": float(cfg.sim.fill_prob),
            "adverse_frac": float(cfg.sim.adverse_frac),
        }
    )
    update["sim"] = sim
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


def default_15m_param_grid() -> list[dict[str, float]]:
    """Compact grid in 15m bar units (≈ same wall-clock as frozen 1h grid).

    1h tau∈{16,32,48} → 15m ×4 → {64,128,192}
    1h vol_lookback=192 → 768; ewma_halflife=96 → 384
    """
    gammas = [0.08, 0.12, 0.18]
    kappas = [0.8, 1.2]
    taus = [64.0, 128.0, 192.0]
    etas = [2.5, 3.5]
    advs = [0.35, 0.45]
    vol_lookback = 768
    ewma_halflife = 384.0
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


def fifo_hold_hours_from_fills(
    frame: pd.DataFrame,
    *,
    hours_per_bar: float = HOURS_PER_BAR,
) -> dict[str, float | None]:
    buy_col = "buy_fill"
    sell_col = "sell_fill"
    buy_sz = "buy_size"
    sell_sz = "sell_size"
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


def calibrate_15m(
    cfg: MarketMakingConfig,
    *,
    is_months: list[tuple[int, int]] | None = None,
    oos_months: list[tuple[int, int]] | None = None,
    timeframe: str = "15m",
    seed: int = 42,
    grid: list[dict[str, float]] | None = None,
    top_k: int = 8,
    portfolio_equity: float = 1000.0,
) -> FifteenMinCalibResult:
    # Match frozen hourly v2 window (Jan–Feb 2026 IS, Mar 2026 OOS).
    is_months = is_months or [(2026, 1), (2026, 2)]
    oos_months = oos_months or [(2026, 3)]
    grid = grid or default_15m_param_grid()

    ranked: list[tuple[tuple[float, float, float], dict[str, float], dict[str, float]]] = []
    for i, params in enumerate(grid):
        trial = _apply_params(cfg, params)
        is_stats = _eval_months(trial, is_months, timeframe=timeframe, seed=seed)
        ranked.append((_score(is_stats), params, is_stats))
        if (i + 1) % 6 == 0:
            print(f"15m calib progress {i + 1}/{len(grid)}", flush=True)

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
        holds = fifo_hold_hours_from_fills(frame, hours_per_bar=HOURS_PER_BAR)
        portfolio_oos[f"{year:04d}-{month:02d}"] = {
            "start_equity": float(pres.start_equity),
            "final_equity": float(pres.final_equity),
            "total_return_pct": float(pres.total_return_pct),
            "n_entries": int(pres.n_entries),
            "n_round_trips": int(pres.n_round_trips),
            "n_skipped_busy": int(pres.n_skipped_busy),
            "fees_paid": float(pres.fees_paid),
            "net_positive": bool(pres.net_positive),
            "holds": holds,
        }

    top_candidates = [
        {"params": p, "is_stats": s, "score": list(sc)} for sc, p, s in ranked[:top_k]
    ]

    return FifteenMinCalibResult(
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
            f"15m candle AS grid (wall-matched to 1h×4); IS={is_months}; OOS={oos_months}; "
            "select by pct_pos→mean_ret→min_ret. Not a live-edge claim."
        ),
    )


def save_15m_calib(
    result: FifteenMinCalibResult,
    *,
    root: str | Path = "data",
    exchange_id: str = "okx",
    basename: str = "fifteen_m_calib_latest",
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
    """Merge selected params into market_making section (15m configs only)."""
    path = Path(path)
    if "15m" not in path.name and "fifteen" not in path.name:
        raise ValueError(
            f"Refusing to write 15m params into non-15m config {path} "
            "(1h package is frozen — see config/LOCKED_HOURLY_V2.md)."
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    mm = dict(raw.get("market_making") or {})
    for k, v in params.items():
        mm[k] = int(v) if k == "vol_lookback" else float(v)
    if "sim" not in mm or not isinstance(mm["sim"], dict):
        mm["sim"] = {}
    mm["sim"]["bars_per_year"] = BARS_PER_YEAR
    raw["market_making"] = mm
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def lock_params_json(
    result: FifteenMinCalibResult,
    *,
    path: str | Path = "config/params_15m_v1_jf2026.json",
    fill_prob: float = 0.25,
    adverse_frac: float = 0.55,
    commission_pct: float = 0.07,
) -> str:
    path = Path(path)
    payload = {
        "label": "fifteen_m_v1_is_202601_202602",
        "created_from": "market_making.calibrate_15m",
        "timeframe": "15m",
        "hours_per_bar": HOURS_PER_BAR,
        "bars_per_year": BARS_PER_YEAR,
        "is_months": [f"{y:04d}-{m:02d}" for y, m in result.is_months],
        "oos_check": [f"{y:04d}-{m:02d}" for y, m in result.oos_months],
        "same_wall_clock_as_hourly_v2": True,
        "interpretation_allowed": False,
        "params": {
            **result.selected_params,
            "commission_pct": float(commission_pct),
            "fill_prob": float(fill_prob),
            "adverse_frac": float(adverse_frac),
            "bars_per_year": BARS_PER_YEAR,
        },
        "is_stats": result.is_stats,
        "oos_stats": result.oos_stats,
        "portfolio_oos": result.portfolio_oos,
        "top_candidates": result.top_candidates,
        "method": result.method,
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    art = Path("/opt/cursor/artifacts")
    art.mkdir(parents=True, exist_ok=True)
    (art / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return str(path)
