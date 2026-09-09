"""Calibrate AS MM params on 1-minute candles (week windows).

Bar knobs are in 1m units; wall-clock targets match frozen 1h (×60) / 5m (×5):
  tau ∈ {960,1920,2880}  (=16/32/48h)
  vol_lookback = 11520   (=192h)
  ewma_halflife = 5760   (=96h)

Default IS: last week of August 2026 (2026-08-25 → 2026-09-01).
OOS check: first week of September 2026 (2026-09-01 → 2026-09-08).
Warmup starts ~8 days before IS so vol_lookback can fill.

Does not modify frozen 1h / 15m / 5m locks.
interpretation_allowed=false.
"""

from __future__ import annotations

import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from market_making.config import MarketMakingConfig
from market_making.simulator import load_okx_mids

HOURS_PER_BAR = 1.0 / 60.0
BARS_PER_YEAR = 525_960.0  # 365.25 * 24 * 60

# Default windows (UTC)
IS_START = "2026-08-25"
IS_END = "2026-09-01"
OOS_START = "2026-09-01"
OOS_END = "2026-09-08"
# ~8d warmup for vol_lookback=11520
WARMUP_START = "2026-08-17"


@dataclass
class OneMinCalibResult:
    selected_params: dict[str, float]
    is_window: dict[str, str]
    oos_window: dict[str, str]
    warmup_start: str
    timeframe: str
    is_stats: dict[str, float]
    oos_stats: dict[str, float]
    top_candidates: list[dict[str, Any]]
    interpretation_allowed: bool
    method: str
    hours_per_bar: float = HOURS_PER_BAR
    bars_per_year: float = BARS_PER_YEAR


def _apply_params(cfg: MarketMakingConfig, params: dict[str, float]) -> MarketMakingConfig:
    update = dict(params)
    if "vol_lookback" in update:
        update["vol_lookback"] = int(update["vol_lookback"])
    sim = cfg.sim.model_copy(
        update={
            "bars_per_year": BARS_PER_YEAR,
            "fill_prob": float(cfg.sim.fill_prob),
            "adverse_frac": float(cfg.sim.adverse_frac),
        }
    )
    update["sim"] = sim
    return cfg.model_copy(update=update)


def run_window_sim(
    cfg: MarketMakingConfig,
    *,
    symbol: str,
    timeframe: str,
    warmup_start: str,
    window_start: str,
    window_end: str,
    seed: int,
) -> dict[str, float]:
    """Run mid-path sim from warmup→end; score equity return only on window.

    Fast path: no per-bar DataFrame (needed for 1m grid search).
    """
    from market_making.engine import QuoteEngine
    from market_making.fees import fee_on_notional
    from market_making.inventory import InventoryState
    from market_making.simulator import _fill_prob

    mids, stamps = load_okx_mids(
        symbol,
        timeframe=timeframe,
        start=warmup_start,
        end_exclusive=window_end,
        root=cfg.storage_root,
        exchange_id=cfg.exchange_id,
    )
    mids = np.asarray(mids, dtype=float)
    stamps = pd.to_datetime(pd.Index(stamps), utc=True)
    mid0 = float(mids[0])
    if mid0 > 0:
        cfg = cfg.model_copy(update={"max_inventory": 10_000.0 / mid0})

    sc = cfg.sim
    eng = QuoteEngine(cfg)
    look = max(int(cfg.vol_lookback), 8)
    inv = InventoryState(
        base=0.0,
        quote=cfg.max_inventory * mid0 * 2.0,
        target=cfg.target_inventory,
        max_abs=cfg.max_inventory,
        halt_frac=cfg.inventory_halt_frac,
    )
    rng = np.random.default_rng(seed + 17)
    t0 = pd.Timestamp(window_start, tz="UTC")
    t1 = pd.Timestamp(window_end, tz="UTC")
    eq0 = eq1 = None
    halfs: list[float] = []
    n_scored = 0

    for i, mid in enumerate(mids):
        mid_f = float(mid)
        lo = 0 if i + 1 <= look else i + 1 - look
        q = eng.quote(mid_f, inv, mid_history=mids[lo : i + 1])
        if q.quoting and q.bid_size > 0:
            if rng.random() < _fill_prob(mid_f, q.bid, q.half_spread_bps, sc.fill_prob):
                px = q.bid * (1.0 + sc.adverse_frac * (q.half_spread_bps / 1e4))
                fee = fee_on_notional(px * q.bid_size, cfg)
                inv.apply_buy(price=px, size=q.bid_size, fee=fee, bar=i)
        if q.quoting and q.ask_size > 0:
            if rng.random() < _fill_prob(mid_f, q.ask, q.half_spread_bps, sc.fill_prob):
                px = q.ask * (1.0 - sc.adverse_frac * (q.half_spread_bps / 1e4))
                fee = fee_on_notional(px * q.ask_size, cfg)
                inv.apply_sell(price=px, size=q.ask_size, fee=fee, bar=i)

        ts = stamps[i]
        if ts >= t0 and ts < t1:
            eq = inv.mark_to_market(mid_f)
            if eq0 is None:
                eq0 = eq
            eq1 = eq
            n_scored += 1
            if q.quoting:
                halfs.append(float(q.half_spread_bps))

    if eq0 is None or eq1 is None or n_scored == 0:
        return {"ret": 0.0, "half": 0.0, "n": 0.0}
    ret = 100.0 * (eq1 - eq0) / eq0 if eq0 else 0.0
    return {
        "ret": float(ret),
        "half": float(np.mean(halfs)) if halfs else 0.0,
        "n": float(n_scored),
    }


def _eval_window(
    cfg: MarketMakingConfig,
    *,
    symbols: list[str],
    timeframe: str,
    warmup_start: str,
    window_start: str,
    window_end: str,
    seed: int,
) -> dict[str, float]:
    rets: list[float] = []
    halfs: list[float] = []
    for sym in symbols:
        try:
            s = run_window_sim(
                cfg,
                symbol=sym,
                timeframe=timeframe,
                warmup_start=warmup_start,
                window_start=window_start,
                window_end=window_end,
                seed=seed,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"  skip {sym}: {exc}", flush=True)
            continue
        rets.append(s["ret"])
        halfs.append(s["half"])
    arr = np.asarray(rets, dtype=float)
    return {
        "n": float(len(arr)),
        "pct_pos": float(100.0 * np.mean(arr > 0)) if len(arr) else 0.0,
        "mean_ret": float(arr.mean()) if len(arr) else 0.0,
        "min_ret": float(arr.min()) if len(arr) else 0.0,
        "mean_half": float(np.mean(halfs)) if halfs else 0.0,
    }


def default_1m_param_grid(*, compact: bool = True) -> list[dict[str, float]]:
    """Grid in 1m bars (wall-matched to frozen 1h ×60).

    compact=True (default): 18 pts — kappa/eta fixed to 5m-lock values for week search.
    compact=False: full 72-pt product grid.
    """
    gammas = [0.08, 0.12, 0.18]
    taus = [960.0, 1920.0, 2880.0]
    advs = [0.35, 0.45]
    vol_lookback = 11520
    ewma_halflife = 5760.0
    if compact:
        kappas = [0.8]
        etas = [2.5]
    else:
        kappas = [0.8, 1.2]
        etas = [2.5, 3.5]
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


def _eval_one_trial(payload: dict[str, Any]) -> tuple[int, dict[str, float], dict[str, float]]:
    cfg = MarketMakingConfig.model_validate(payload["cfg"])
    params = payload["params"]
    idx = int(payload["idx"])
    trial = _apply_params(cfg, params)
    is_stats = _eval_window(
        trial,
        symbols=list(payload["symbols"]),
        timeframe=str(payload["timeframe"]),
        warmup_start=str(payload["warmup_start"]),
        window_start=str(payload["is_start"]),
        window_end=str(payload["is_end"]),
        seed=int(payload["seed"]),
    )
    return idx, params, is_stats


def calibrate_1m(
    cfg: MarketMakingConfig,
    *,
    is_start: str = IS_START,
    is_end: str = IS_END,
    oos_start: str = OOS_START,
    oos_end: str = OOS_END,
    warmup_start: str = WARMUP_START,
    timeframe: str = "1m",
    seed: int = 42,
    grid: list[dict[str, float]] | None = None,
    top_k: int = 8,
    n_workers: int | None = None,
    calib_symbols: list[str] | None = None,
    compact_grid: bool = True,
) -> OneMinCalibResult:
    grid = grid or default_1m_param_grid(compact=compact_grid)
    symbols = list(calib_symbols or cfg.symbols)
    workers = n_workers if n_workers is not None else max(1, min(4, (os.cpu_count() or 2)))

    ranked: list[tuple[tuple[float, float, float], dict[str, float], dict[str, float]]] = []
    cfg_dump = cfg.model_dump()
    payloads = [
        {
            "idx": i,
            "cfg": cfg_dump,
            "params": params,
            "symbols": symbols,
            "timeframe": timeframe,
            "warmup_start": warmup_start,
            "is_start": is_start,
            "is_end": is_end,
            "seed": seed,
        }
        for i, params in enumerate(grid)
    ]

    print(
        f"1m calib IS {is_start}→{is_end} (warmup {warmup_start}) "
        f"symbols={len(symbols)} grid={len(grid)} workers={workers}",
        flush=True,
    )
    if workers <= 1:
        for i, payload in enumerate(payloads):
            idx, params, is_stats = _eval_one_trial(payload)
            ranked.append((_score(is_stats), params, is_stats))
            print(f"1m calib progress {i + 1}/{len(grid)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_eval_one_trial, p): p["idx"] for p in payloads}
            done = 0
            for fut in as_completed(futs):
                idx, params, is_stats = fut.result()
                ranked.append((_score(is_stats), params, is_stats))
                done += 1
                print(f"1m calib progress {done}/{len(grid)}", flush=True)

    ranked.sort(key=lambda t: t[0], reverse=True)
    best_params = ranked[0][1]
    best_cfg = _apply_params(cfg, best_params)
    is_stats = ranked[0][2]
    print(f"1m calib OOS {oos_start}→{oos_end} ...", flush=True)
    oos_stats = _eval_window(
        best_cfg,
        symbols=symbols,
        timeframe=timeframe,
        warmup_start=warmup_start,
        window_start=oos_start,
        window_end=oos_end,
        seed=seed,
    )
    top = [
        {"params": p, "is_stats": s, "score": list(sc)}
        for sc, p, s in ranked[:top_k]
    ]
    return OneMinCalibResult(
        selected_params=best_params,
        is_window={"start": is_start, "end_exclusive": is_end},
        oos_window={"start": oos_start, "end_exclusive": oos_end},
        warmup_start=warmup_start,
        timeframe=timeframe,
        is_stats=is_stats,
        oos_stats=oos_stats,
        top_candidates=top,
        interpretation_allowed=False,
        method=(
            "1m candle AS grid (wall-matched to 1h×60); "
            f"IS={is_start}→{is_end}; OOS={oos_start}→{oos_end}; "
            "select max pct_pos then mean_ret then min_ret"
        ),
    )


def save_1m_calib(result: OneMinCalibResult, path: str | Path) -> None:
    Path(path).write_text(json.dumps(asdict(result), indent=2, default=str))


def write_params_to_yaml(params: dict[str, float], path: str | Path) -> None:
    path = Path(path)
    if "1m" not in path.name and "one" not in path.name:
        raise ValueError(
            f"Refusing to write 1m params into non-1m config {path} "
            "(protects frozen 5m/15m/1h packages)"
        )
    raw = yaml.safe_load(path.read_text()) or {}
    mm = raw.setdefault("market_making", {})
    for k, v in params.items():
        if k == "vol_lookback":
            mm[k] = int(v)
        else:
            mm[k] = float(v) if isinstance(v, (int, float)) else v
    mm.setdefault("sim", {})
    mm["sim"]["bars_per_year"] = BARS_PER_YEAR
    path.write_text(yaml.safe_dump(raw, sort_keys=False))


def lock_params_json(
    result: OneMinCalibResult,
    path: str | Path = "config/params_1m_v1_auglastweek2026.json",
) -> None:
    payload = {
        "locked": True,
        "interpretation_allowed": False,
        "created_from": "market_making.calibrate_1m",
        "timeframe": "1m",
        "hours_per_bar": HOURS_PER_BAR,
        "bars_per_year": BARS_PER_YEAR,
        "is_window": result.is_window,
        "oos_window": result.oos_window,
        "warmup_start": result.warmup_start,
        "selected_params": result.selected_params,
        "is_stats": result.is_stats,
        "oos_stats": result.oos_stats,
        "method": result.method,
        "sim": {
            "fill_prob": 0.25,
            "adverse_frac": 0.55,
            "bars_per_year": BARS_PER_YEAR,
        },
    }
    Path(path).write_text(json.dumps(payload, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    import argparse

    from market_making.config import load_mm_config

    p = argparse.ArgumentParser(description="Calibrate 1m AS MM on Aug last week")
    p.add_argument("--config", default="config/market_making_1m.yaml")
    p.add_argument("--is-start", default=IS_START)
    p.add_argument("--is-end", default=IS_END)
    p.add_argument("--oos-start", default=OOS_START)
    p.add_argument("--oos-end", default=OOS_END)
    p.add_argument("--warmup-start", default=WARMUP_START)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-symbols", type=int, default=30)
    p.add_argument("--full-grid", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lock-path", default="config/params_1m_v1_auglastweek2026.json")
    args = p.parse_args(argv)

    cfg = load_mm_config(args.config)
    symbols = list(cfg.symbols)[: max(2, int(args.max_symbols))]
    result = calibrate_1m(
        cfg,
        is_start=args.is_start,
        is_end=args.is_end,
        oos_start=args.oos_start,
        oos_end=args.oos_end,
        warmup_start=args.warmup_start,
        seed=int(args.seed),
        n_workers=int(args.workers),
        calib_symbols=symbols,
        compact_grid=not bool(args.full_grid),
    )
    save_1m_calib(result, "config/calib_1m_auglastweek2026_full.json")
    lock_params_json(result, args.lock_path)
    write_params_to_yaml(result.selected_params, args.config)
    print(
        f"LOCKED {args.lock_path}: {result.selected_params}\n"
        f"  IS pct_pos={result.is_stats['pct_pos']:.1f}% mean={result.is_stats['mean_ret']:.3f}%\n"
        f"  OOS pct_pos={result.oos_stats['pct_pos']:.1f}% mean={result.oos_stats['mean_ret']:.3f}%",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
