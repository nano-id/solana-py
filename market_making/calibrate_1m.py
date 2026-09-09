"""Calibrate AS MM params on 1-minute candles with OHLC touch fills.

v2 (OHLC-touch, 1m-native horizons)
------------------------------------
- Fill model: deterministic — buy if candle low <= bid, sell if high >= ask.
  No Bernoulli fill_prob / adverse_frac dice.
- Time knobs are *1m-scale*, not wall-matched 1h×60:
    tau ∈ {30,60,120,240} bars (=0.5–4h)
    vol_lookback ∈ {180,360,720} (=3–12h)
    ewma_halflife = lookback / 2
- Same IS/OOS week windows as v1 lock.
- Selection: max pct_pos → mean_ret → min_ret (under touch fills).

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

HOURS_PER_BAR = 1.0 / 60.0
BARS_PER_YEAR = 525_960.0  # 365.25 * 24 * 60

IS_START = "2026-08-25"
IS_END = "2026-09-01"
OOS_START = "2026-09-01"
OOS_END = "2026-09-08"
# Max lookback 720 → ≥12h warmup; keep Aug-17 for headroom / data continuity.
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
    fill_model: str = "ohlc_touch"


def _apply_params(cfg: MarketMakingConfig, params: dict[str, float]) -> MarketMakingConfig:
    update = dict(params)
    if "vol_lookback" in update:
        update["vol_lookback"] = int(update["vol_lookback"])
    # Touch model: fill_prob unused; keep bars_per_year correct for any downstream.
    sim = cfg.sim.model_copy(
        update={
            "bars_per_year": BARS_PER_YEAR,
            "fill_prob": 1.0,  # unused under touch; document as N/A in lock
            "adverse_frac": 0.0,
        }
    )
    update["sim"] = sim
    return cfg.model_copy(update=update)


def _load_ohlc(
    symbol: str,
    *,
    timeframe: str,
    start: str,
    end_exclusive: str,
    root: str | Path,
    exchange_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    path = Path(root) / exchange_id / timeframe / f"{symbol.replace('/', '_')}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    mask = (ts >= pd.Timestamp(start, tz="UTC")) & (ts < pd.Timestamp(end_exclusive, tz="UTC"))
    sl = df.loc[mask].sort_values("timestamp")
    if len(sl) < 2:
        raise ValueError(f"Insufficient bars for {symbol} {start}→{end_exclusive}: {len(sl)}")
    stamps = pd.DatetimeIndex(pd.to_datetime(sl["timestamp"], unit="ms", utc=True))
    return (
        sl["open"].to_numpy(dtype=float),
        sl["high"].to_numpy(dtype=float),
        sl["low"].to_numpy(dtype=float),
        sl["close"].to_numpy(dtype=float),
        stamps,
    )


def run_window_sim(
    cfg: MarketMakingConfig,
    *,
    symbol: str,
    timeframe: str,
    warmup_start: str,
    window_start: str,
    window_end: str,
    seed: int = 0,
) -> dict[str, float]:
    """OHLC-touch mid-path AS sim; score equity return only on [window_start, window_end).

    seed retained for API compatibility (unused — deterministic).
    """
    del seed  # deterministic
    from market_making.engine import QuoteEngine
    from market_making.fees import fee_on_notional
    from market_making.inventory import InventoryState

    _o, highs, lows, closes, stamps = _load_ohlc(
        symbol,
        timeframe=timeframe,
        start=warmup_start,
        end_exclusive=window_end,
        root=cfg.storage_root,
        exchange_id=cfg.exchange_id,
    )
    mid0 = float(closes[0])
    if mid0 > 0:
        cfg = cfg.model_copy(update={"max_inventory": 10_000.0 / mid0})

    eng = QuoteEngine(cfg)
    look = max(int(cfg.vol_lookback), 8)
    inv = InventoryState(
        base=0.0,
        quote=cfg.max_inventory * mid0 * 2.0,
        target=cfg.target_inventory,
        max_abs=cfg.max_inventory,
        halt_frac=cfg.inventory_halt_frac,
    )
    t0 = pd.Timestamp(window_start, tz="UTC")
    t1 = pd.Timestamp(window_end, tz="UTC")
    eq0 = eq1 = None
    halfs: list[float] = []
    n_scored = 0
    n_buys = n_sells = 0

    for i, mid in enumerate(closes):
        mid_f = float(mid)
        lo = 0 if i + 1 <= look else i + 1 - look
        q = eng.quote(mid_f, inv, mid_history=closes[lo : i + 1])
        hi_i = float(highs[i])
        lo_i = float(lows[i])

        # Deterministic touch fills at quote (no adverse dice).
        if q.quoting and q.bid_size > 0 and lo_i <= q.bid + 1e-12:
            px = float(q.bid)
            fee = fee_on_notional(px * q.bid_size, cfg)
            inv.apply_buy(price=px, size=q.bid_size, fee=fee, bar=i)
            n_buys += 1
        if q.quoting and q.ask_size > 0 and hi_i >= q.ask - 1e-12:
            # Cap sell to inventory (spot long-only book).
            sell_sz = min(float(q.ask_size), max(float(inv.base), 0.0))
            if sell_sz > 1e-12:
                px = float(q.ask)
                fee = fee_on_notional(px * sell_sz, cfg)
                inv.apply_sell(price=px, size=sell_sz, fee=fee, bar=i)
                n_sells += 1

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
        return {
            "ret": 0.0,
            "half": 0.0,
            "n": 0.0,
            "n_buys": 0.0,
            "n_sells": 0.0,
        }
    ret = 100.0 * (eq1 - eq0) / eq0 if eq0 else 0.0
    return {
        "ret": float(ret),
        "half": float(np.mean(halfs)) if halfs else 0.0,
        "n": float(n_scored),
        "n_buys": float(n_buys),
        "n_sells": float(n_sells),
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
    buys: list[float] = []
    sells: list[float] = []
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
        buys.append(s["n_buys"])
        sells.append(s["n_sells"])
    arr = np.asarray(rets, dtype=float)
    return {
        "n": float(len(arr)),
        "pct_pos": float(100.0 * np.mean(arr > 0)) if len(arr) else 0.0,
        "mean_ret": float(arr.mean()) if len(arr) else 0.0,
        "min_ret": float(arr.min()) if len(arr) else 0.0,
        "mean_half": float(np.mean(halfs)) if halfs else 0.0,
        "mean_buys": float(np.mean(buys)) if buys else 0.0,
        "mean_sells": float(np.mean(sells)) if sells else 0.0,
    }


def default_1m_param_grid(*, compact: bool = True) -> list[dict[str, float]]:
    """1m-native grid (NOT 1h×60 wall-match).

    compact: 36 pts. full: 72 pts (kappa/eta expanded).
    """
    gammas = [0.08, 0.12, 0.18]
    taus = [30.0, 60.0, 120.0, 240.0]  # 0.5h–4h
    advs = [0.25, 0.35]
    lookbacks = [180.0, 360.0, 720.0]  # 3h / 6h / 12h
    if compact:
        # Drop longest lookback to keep week search tractable → 24 pts
        lookbacks = [180.0, 360.0]
        kappas = [0.8]
        etas = [2.5]
        liqs = [4.0]
    else:
        kappas = [0.8, 1.2]
        etas = [2.5]
        liqs = [4.0, 6.0]
    out: list[dict[str, float]] = []
    for g, k, t, e, a, vl, lq in itertools.product(
        gammas, kappas, taus, etas, advs, lookbacks, liqs
    ):
        out.append(
            {
                "gamma": float(g),
                "kappa": float(k),
                "tau_bars": float(t),
                "size_eta": float(e),
                "adverse_vol_mult": float(a),
                "vol_lookback": float(vl),
                "ewma_halflife": float(vl / 2.0),
                "liq_bps": float(lq),
                "vol_spread_mult": 0.25,
                "inventory_halt_frac": 0.80,
            }
        )
    return out


def _score(is_stats: dict[str, float]) -> tuple[float, float, float, float]:
    # Prefer profitable + active books (mean_sells as activity proxy).
    return (
        is_stats["pct_pos"],
        is_stats["mean_ret"],
        is_stats["min_ret"],
        float(is_stats.get("mean_sells", 0.0)),
    )


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

    ranked: list[tuple[tuple[float, float, float, float], dict[str, float], dict[str, float]]] = []
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
        f"1m OHLC-touch calib IS {is_start}→{is_end} (warmup {warmup_start}) "
        f"symbols={len(symbols)} grid={len(grid)} workers={workers}",
        flush=True,
    )
    if workers <= 1:
        for i, payload in enumerate(payloads):
            idx, params, is_stats = _eval_one_trial(payload)
            ranked.append((_score(is_stats), params, is_stats))
            print(
                f"1m calib progress {i + 1}/{len(grid)} "
                f"pct={is_stats['pct_pos']:.0f} mean={is_stats['mean_ret']:.3f} "
                f"half={is_stats['mean_half']:.1f}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_eval_one_trial, p): p["idx"] for p in payloads}
            done = 0
            for fut in as_completed(futs):
                idx, params, is_stats = fut.result()
                ranked.append((_score(is_stats), params, is_stats))
                done += 1
                print(
                    f"1m calib progress {done}/{len(grid)} "
                    f"pct={is_stats['pct_pos']:.0f} mean={is_stats['mean_ret']:.3f} "
                    f"half={is_stats['mean_half']:.1f}",
                    flush=True,
                )

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
            "1m OHLC-touch AS grid (1m-native tau/lookback; NOT 1h×60); "
            f"IS={is_start}→{is_end}; OOS={oos_start}→{oos_end}; "
            "select max pct_pos then mean_ret then min_ret then mean_sells"
        ),
        fill_model="ohlc_touch",
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
    mm["sim"]["fill_prob"] = 1.0  # unused under OHLC touch
    mm["sim"]["adverse_frac"] = 0.0
    path.write_text(yaml.safe_dump(raw, sort_keys=False))


def lock_params_json(
    result: OneMinCalibResult,
    path: str | Path = "config/params_1m_v2_ohlc_touch_auglastweek2026.json",
) -> None:
    payload = {
        "locked": True,
        "interpretation_allowed": False,
        "created_from": "market_making.calibrate_1m",
        "timeframe": "1m",
        "hours_per_bar": HOURS_PER_BAR,
        "bars_per_year": BARS_PER_YEAR,
        "fill_model": "ohlc_touch",
        "is_window": result.is_window,
        "oos_window": result.oos_window,
        "warmup_start": result.warmup_start,
        "selected_params": result.selected_params,
        "is_stats": result.is_stats,
        "oos_stats": result.oos_stats,
        "method": result.method,
        "sim": {
            "fill_model": "ohlc_touch",
            "fill_prob": None,
            "adverse_frac": 0.0,
            "bars_per_year": BARS_PER_YEAR,
        },
        "supersedes": "config/params_1m_v1_auglastweek2026.json",
        "notes": (
            "v2 replaces 1h×60 wall-matched knobs + Bernoulli fills with "
            "1m-native horizons and deterministic OHLC touch fills."
        ),
    }
    Path(path).write_text(json.dumps(payload, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    import argparse

    from market_making.config import load_mm_config

    p = argparse.ArgumentParser(description="Calibrate 1m AS MM with OHLC touch fills")
    p.add_argument("--config", default="config/market_making_1m_fee008.yaml")
    p.add_argument("--is-start", default=IS_START)
    p.add_argument("--is-end", default=IS_END)
    p.add_argument("--oos-start", default=OOS_START)
    p.add_argument("--oos-end", default=OOS_END)
    p.add_argument("--warmup-start", default=WARMUP_START)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-symbols", type=int, default=30)
    p.add_argument("--full-grid", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--lock-path",
        default="config/params_1m_v2_ohlc_touch_auglastweek2026.json",
    )
    p.add_argument(
        "--calib-out",
        default="config/calib_1m_v2_ohlc_touch_auglastweek2026_full.json",
    )
    p.add_argument(
        "--write-yamls",
        nargs="*",
        default=[
            "config/market_making_1m_fee008.yaml",
            "config/market_making_1m.yaml",
            "config/live_okx_tr_1m_grow150.yaml",
        ],
        help="1m yaml configs to update with selected params",
    )
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
    save_1m_calib(result, args.calib_out)
    lock_params_json(result, args.lock_path)
    for yml in args.write_yamls:
        if Path(yml).exists():
            write_params_to_yaml(result.selected_params, yml)
            print(f"updated {yml}", flush=True)
    print(
        f"LOCKED {args.lock_path}: {result.selected_params}\n"
        f"  IS pct_pos={result.is_stats['pct_pos']:.1f}% mean={result.is_stats['mean_ret']:.3f}% "
        f"half={result.is_stats['mean_half']:.1f} buys={result.is_stats.get('mean_buys', 0):.0f}\n"
        f"  OOS pct_pos={result.oos_stats['pct_pos']:.1f}% mean={result.oos_stats['mean_ret']:.3f}% "
        f"half={result.oos_stats['mean_half']:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
