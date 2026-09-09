"""Realism tests for 1m MM on majors: slippage, regime stability, fee vs edge.

Measures (paper / observation only — no live orders):
1. Live BBO spreads on majors (slippage proxy)
2. 1-minute mid moves vs round-trip fee
3. Hist regime persistence (vol regime flips: minutes vs days)
4. AS quote net edge vs fee on recent 1m path

interpretation_allowed=false.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from market_making.config import load_mm_config
from market_making.engine import QuoteEngine
from market_making.feed import connect_spot_exchange, snapshot_from_order_book
from market_making.inventory import InventoryState
from market_making.simulator import load_okx_mids

MAJORS_DEFAULT = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "XRP/USDT",
]


@dataclass
class RealismReport:
    interpretation_allowed: bool
    symbols: list[str]
    fee_bps_per_side: float
    fee_bps_roundtrip: float
    live: dict[str, Any]
    hist_regime: dict[str, Any]
    fee_vs_move: dict[str, Any]
    as_edge_sample: dict[str, Any]
    verdicts: dict[str, str]
    notes: str


def _poll_spreads(
    symbols: list[str],
    *,
    exchange_id: str,
    duration_sec: float,
    poll_sec: float,
) -> dict[str, Any]:
    connected = connect_spot_exchange(exchange_id)
    ex = connected.exchange
    eid = connected.exchange_id
    t0 = time.time()
    rows: list[dict[str, Any]] = []
    last_mid: dict[str, float] = {}
    print(
        f"Live BBO spread probe: {len(symbols)} majors, {duration_sec:.0f}s @ {poll_sec}s",
        flush=True,
    )
    while time.time() - t0 < duration_sec:
        now_ms = int(time.time() * 1000)
        for sym in symbols:
            try:
                book = ex.fetch_order_book(sym, limit=5)
            except Exception as exc:  # noqa: BLE001
                print(f"  book fail {sym}: {exc}", flush=True)
                continue
            snap = snapshot_from_order_book(sym, book, exchange_id=eid, ts_ms=now_ms)
            if snap is None or not snap.valid:
                continue
            dmid_bps = None
            if sym in last_mid and last_mid[sym] > 0:
                dmid_bps = 1e4 * (snap.mid - last_mid[sym]) / last_mid[sym]
            last_mid[sym] = snap.mid
            rows.append(
                {
                    "ts_ms": snap.ts_ms,
                    "symbol": sym,
                    "mid": snap.mid,
                    "spread_bps": snap.spread_bps,
                    "dmid_bps": dmid_bps,
                }
            )
        time.sleep(max(poll_sec, 0.2))
        if int(time.time() - t0) % 60 < poll_sec:
            print(f"  … live {time.time() - t0:.0f}s n={len(rows)}", flush=True)

    if not rows:
        return {"n": 0, "error": "no_books"}
    df = pd.DataFrame(rows)
    by = {}
    for sym, g in df.groupby("symbol"):
        sp = g["spread_bps"].astype(float)
        dm = g["dmid_bps"].dropna().astype(float)
        by[sym] = {
            "n": int(len(g)),
            "spread_bps_mean": float(sp.mean()),
            "spread_bps_median": float(sp.median()),
            "spread_bps_p90": float(sp.quantile(0.9)),
            "spread_bps_max": float(sp.max()),
            "abs_dmid_bps_mean": float(dm.abs().mean()) if len(dm) else None,
            "abs_dmid_bps_p90": float(dm.abs().quantile(0.9)) if len(dm) else None,
        }
    all_sp = df["spread_bps"].astype(float)
    return {
        "n_samples": int(len(df)),
        "duration_sec": float(time.time() - t0),
        "spread_bps_mean": float(all_sp.mean()),
        "spread_bps_median": float(all_sp.median()),
        "spread_bps_p90": float(all_sp.quantile(0.9)),
        "by_symbol": by,
    }


def _hist_regime(
    symbols: list[str],
    *,
    start: str,
    end_exclusive: str,
    root: str,
    exchange_id: str,
    vol_lookback: int = 60,
) -> dict[str, Any]:
    """Vol persistence on 1m: flip gaps + autocorrelation at 1h/6h/1d lags."""
    flip_minutes: list[float] = []
    n_flips = 0
    n_bars_total = 0
    ac_1h: list[float] = []
    ac_6h: list[float] = []
    ac_1d: list[float] = []
    large_shift_hours: list[float] = []
    per: dict[str, Any] = {}
    for sym in symbols:
        try:
            mids, stamps = load_okx_mids(
                sym,
                timeframe="1m",
                start=start,
                end_exclusive=end_exclusive,
                root=root,
                exchange_id=exchange_id,
            )
        except Exception as exc:  # noqa: BLE001
            per[sym] = {"error": str(exc)}
            continue
        mids = np.asarray(mids, dtype=float)
        if len(mids) < vol_lookback + 100:
            per[sym] = {"error": "short"}
            continue
        rets = np.diff(np.log(np.clip(mids, 1e-12, None)))
        vol = pd.Series(rets).rolling(vol_lookback, min_periods=vol_lookback).std().to_numpy()
        valid = ~np.isnan(vol)
        v = vol[valid]
        if len(v) < 200:
            per[sym] = {"error": "short_vol"}
            continue
        med = float(np.median(v))
        regime = (v > med).astype(int)
        flips = np.where(np.diff(regime) != 0)[0]
        n_flips += int(len(flips))
        n_bars_total += int(len(regime))
        if len(flips) >= 2:
            gaps = np.diff(flips).astype(float)
            flip_minutes.extend(gaps.tolist())

        s = pd.Series(v)
        for lag, bucket in ((60, ac_1h), (360, ac_6h), (1440, ac_1d)):
            if len(s) > lag + 10:
                bucket.append(float(s.autocorr(lag=lag)))

        # time until |vol/vol0 - 1| > 50%
        step = 30
        for i in range(0, len(v) - 60, step):
            v0 = float(v[i])
            if v0 <= 0:
                continue
            hit = None
            for j in range(i + 1, len(v)):
                if abs(v[j] / v0 - 1.0) >= 0.5:
                    hit = j - i
                    break
            if hit is not None:
                large_shift_hours.append(hit / 60.0)

        per[sym] = {
            "n_bars": int(len(regime)),
            "n_flips": int(len(flips)),
            "median_gap_min": float(np.median(np.diff(flips))) if len(flips) >= 2 else None,
            "mean_gap_hours": float(np.mean(np.diff(flips)) / 60.0) if len(flips) >= 2 else None,
            "pct_high_vol": float(100.0 * regime.mean()),
            "vol_autocorr_1h": float(s.autocorr(lag=60)) if len(s) > 70 else None,
            "vol_autocorr_1d": float(s.autocorr(lag=1440)) if len(s) > 1450 else None,
        }
    gaps_m = np.asarray(flip_minutes, dtype=float) if flip_minutes else np.asarray([])
    shift_h = np.asarray(large_shift_hours, dtype=float) if large_shift_hours else np.asarray([])
    return {
        "window": f"{start}→{end_exclusive}",
        "vol_lookback_bars": vol_lookback,
        "n_flips_total": n_flips,
        "n_bars_total": n_bars_total,
        "binary_flip_gap_min_median": float(np.median(gaps_m)) if len(gaps_m) else None,
        "binary_flip_gap_hours_median": float(np.median(gaps_m) / 60.0) if len(gaps_m) else None,
        "pct_binary_gaps_lt_60_min": float(100.0 * np.mean(gaps_m < 60)) if len(gaps_m) else None,
        "vol_autocorr_1h_mean": float(np.mean(ac_1h)) if ac_1h else None,
        "vol_autocorr_6h_mean": float(np.mean(ac_6h)) if ac_6h else None,
        "vol_autocorr_1d_mean": float(np.mean(ac_1d)) if ac_1d else None,
        "hours_to_50pct_vol_change_median": float(np.median(shift_h)) if len(shift_h) else None,
        "hours_to_50pct_vol_change_p25": float(np.percentile(shift_h, 25)) if len(shift_h) else None,
        "pct_50pct_vol_change_ge_24h": float(100.0 * np.mean(shift_h >= 24)) if len(shift_h) else None,
        "by_symbol": per,
    }


def _fee_vs_1m_moves(
    symbols: list[str],
    *,
    start: str,
    end_exclusive: str,
    root: str,
    exchange_id: str,
    fee_rt_bps: float,
) -> dict[str, Any]:
    abs_1m: list[float] = []
    per: dict[str, Any] = {}
    for sym in symbols:
        try:
            mids, _ = load_okx_mids(
                sym,
                timeframe="1m",
                start=start,
                end_exclusive=end_exclusive,
                root=root,
                exchange_id=exchange_id,
            )
        except Exception as exc:  # noqa: BLE001
            per[sym] = {"error": str(exc)}
            continue
        mids = np.asarray(mids, dtype=float)
        moves = 1e4 * np.abs(np.diff(mids) / mids[:-1])
        abs_1m.extend(moves.tolist())
        per[sym] = {
            "n": int(len(moves)),
            "abs_1m_bps_median": float(np.median(moves)),
            "abs_1m_bps_mean": float(np.mean(moves)),
            "pct_1m_move_lt_fee_rt": float(100.0 * np.mean(moves < fee_rt_bps)),
        }
    arr = np.asarray(abs_1m, dtype=float)
    return {
        "fee_bps_roundtrip": fee_rt_bps,
        "abs_1m_bps_median": float(np.median(arr)) if len(arr) else None,
        "abs_1m_bps_mean": float(np.mean(arr)) if len(arr) else None,
        "pct_1m_moves_smaller_than_fee_rt": float(100.0 * np.mean(arr < fee_rt_bps))
        if len(arr)
        else None,
        "by_symbol": per,
    }


def _as_edge_sample(cfg, symbols: list[str], *, start: str, end: str) -> dict[str, Any]:
    edges: list[float] = []
    halves: list[float] = []
    quoting = 0
    total = 0
    for sym in symbols:
        try:
            mids, _ = load_okx_mids(
                sym,
                timeframe="1m",
                start=start,
                end_exclusive=end,
                root=cfg.storage_root,
                exchange_id=cfg.exchange_id,
            )
        except Exception:
            continue
        mids = np.asarray(mids, dtype=float)
        mid0 = float(mids[0])
        max_inv = 10_000.0 / mid0 if mid0 > 0 else 1.0
        local_cfg = cfg.model_copy(update={"max_inventory": max_inv})
        eng = QuoteEngine(local_cfg)
        inv = InventoryState(
            base=0.0, quote=0.0, target=0.0, max_abs=max_inv, halt_frac=cfg.inventory_halt_frac
        )
        look = max(int(cfg.vol_lookback), 8)
        # sample last up to 500 bars
        for i in range(max(look, len(mids) - 500), len(mids)):
            lo = 0 if i + 1 <= look else i + 1 - look
            q = eng.quote(float(mids[i]), inv, mid_history=mids[lo : i + 1])
            total += 1
            if q.quoting:
                quoting += 1
                edges.append(float(q.net_edge_bps_if_matched))
                halves.append(float(q.half_spread_bps))
    e = np.asarray(edges, dtype=float) if edges else np.asarray([])
    h = np.asarray(halves, dtype=float) if halves else np.asarray([])
    return {
        "n_quotes_checked": total,
        "pct_quoting": float(100.0 * quoting / total) if total else 0.0,
        "net_edge_bps_median": float(np.median(e)) if len(e) else None,
        "net_edge_bps_mean": float(np.mean(e)) if len(e) else None,
        "pct_net_edge_positive": float(100.0 * np.mean(e > 0)) if len(e) else None,
        "half_spread_bps_median": float(np.median(h)) if len(h) else None,
        "required_half_spread_bps": float(cfg.required_half_spread_bps),
    }


def build_verdicts(
    *,
    live: dict[str, Any],
    regime: dict[str, Any],
    fee_moves: dict[str, Any],
    edge: dict[str, Any],
    fee_rt: float,
) -> dict[str, str]:
    v: dict[str, str] = {}
    sp = live.get("spread_bps_median")
    if sp is not None:
        if sp < 2.0:
            v["slippage_majors"] = (
                f"SUPPORTED: live median spread {sp:.2f} bps << fee RT {fee_rt:.1f} bps "
                "(majors; slippage/spread drag small vs fee)"
            )
        elif sp < fee_rt:
            v["slippage_majors"] = (
                f"PARTLY: median spread {sp:.2f} bps still below fee RT {fee_rt:.1f}, "
                "but not negligible"
            )
        else:
            v["slippage_majors"] = (
                f"WEAK: median spread {sp:.2f} bps >= fee RT {fee_rt:.1f}"
            )
    gap_h = regime.get("binary_flip_gap_hours_median")
    pct_lt_hour = regime.get("pct_binary_gaps_lt_60_min")
    ac1d = regime.get("vol_autocorr_1d_mean")
    hrs50 = regime.get("hours_to_50pct_vol_change_median")
    pct24 = regime.get("pct_50pct_vol_change_ge_24h")
    if ac1d is not None or hrs50 is not None:
        if (ac1d is not None and ac1d >= 0.5) or (
            hrs50 is not None and hrs50 >= 12
        ) or (pct24 is not None and pct24 >= 40):
            v["regime_stability"] = (
                f"SUPPORTED for slow vol: 1d vol autocorr={ac1d}; "
                f"median hours to ±50% vol change={hrs50}; "
                f"% of paths needing ≥24h={pct24}. "
                f"(Binary high/low flips are noisier: med gap {gap_h}h, <1h={pct_lt_hour}%)"
            )
        elif (ac1d is not None and ac1d >= 0.2) or (hrs50 is not None and hrs50 >= 3):
            v["regime_stability"] = (
                f"MIXED: 1d vol autocorr={ac1d}, median hours to ±50% vol={hrs50}; "
                f"binary flip med {gap_h}h — some persistence, not strictly 'days only'"
            )
        else:
            v["regime_stability"] = (
                f"WEAK vs 'days': 1d autocorr={ac1d}, hours_to_50%_vol={hrs50}, "
                f"binary flip med {gap_h}h"
            )
    pct_small = fee_moves.get("pct_1m_moves_smaller_than_fee_rt")
    if pct_small is not None:
        v["minute_noise_vs_fee"] = (
            f"NOTE: {pct_small:.0f}% of |1m mid moves| < fee RT {fee_rt:.1f} bps "
            "— on 1m, fee dominates raw minute noise; edge must come from spread capture, "
            "not directional 1m moves"
        )
    if edge.get("pct_net_edge_positive") is not None:
        v["as_quote_edge"] = (
            f"AS quotes net_edge>0 on {edge['pct_net_edge_positive']:.0f}% of sampled bars; "
            f"median net_edge={edge['net_edge_bps_median']:.1f} bps "
            f"(half={edge['half_spread_bps_median']:.1f})"
        )
    return v


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="1m majors realism: slip/regime/fee tests")
    p.add_argument("--config", default="config/market_making_1m_live12.yaml")
    p.add_argument("--live-sec", type=float, default=900.0)
    p.add_argument("--poll-sec", type=float, default=2.0)
    p.add_argument("--hist-start", default="2026-08-17")
    p.add_argument("--hist-end", default="2026-09-08")
    p.add_argument("--max-symbols", type=int, default=5)
    p.add_argument("--out", default="config/realism_1m_majors_results.json")
    p.add_argument("--skip-live", action="store_true")
    args = p.parse_args(argv)

    cfg = load_mm_config(args.config)
    symbols = [s for s in MAJORS_DEFAULT if s in cfg.symbols][: int(args.max_symbols)]
    if not symbols:
        symbols = list(cfg.symbols)[: int(args.max_symbols)]
    fee_side = float(cfg.commission_bps)
    fee_rt = 2.0 * fee_side

    print(f"Realism majors={symbols} fee_side={fee_side:.1f}bps RT={fee_rt:.1f}bps", flush=True)

    if args.skip_live:
        live = {"n_samples": 0, "skipped": True}
    else:
        live = _poll_spreads(
            symbols,
            exchange_id=cfg.exchange_id,
            duration_sec=float(args.live_sec),
            poll_sec=float(args.poll_sec),
        )
        Path("/opt/cursor/artifacts").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {"symbol": s, **v}
                for s, v in (live.get("by_symbol") or {}).items()
                if isinstance(v, dict)
            ]
        ).to_csv("/opt/cursor/artifacts/realism_1m_live_spreads.csv", index=False)

    print("Hist regime + fee-vs-move ...", flush=True)
    regime = _hist_regime(
        symbols,
        start=args.hist_start,
        end_exclusive=args.hist_end,
        root=str(cfg.storage_root),
        exchange_id=cfg.exchange_id,
        vol_lookback=60,
    )
    fee_moves = _fee_vs_1m_moves(
        symbols,
        start=args.hist_start,
        end_exclusive=args.hist_end,
        root=str(cfg.storage_root),
        exchange_id=cfg.exchange_id,
        fee_rt_bps=fee_rt,
    )
    edge = _as_edge_sample(
        cfg, symbols, start=args.hist_start, end=args.hist_end
    )
    verdicts = build_verdicts(
        live=live, regime=regime, fee_moves=fee_moves, edge=edge, fee_rt=fee_rt
    )
    report = RealismReport(
        interpretation_allowed=False,
        symbols=symbols,
        fee_bps_per_side=fee_side,
        fee_bps_roundtrip=fee_rt,
        live=live,
        hist_regime=regime,
        fee_vs_move=fee_moves,
        as_edge_sample=edge,
        verdicts=verdicts,
        notes=(
            "Observation only. Slippage proxied by live BBO spread; regime = 1m vol "
            "high/low vs median; no live orders."
        ),
    )
    payload = asdict(report)
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
    art = Path("/opt/cursor/artifacts") / Path(args.out).name
    try:
        art.parent.mkdir(parents=True, exist_ok=True)
        art.write_text(json.dumps(payload, indent=2, default=str))
    except OSError:
        pass

    print(json.dumps(verdicts, indent=2), flush=True)
    print(
        f"live_spread_med={live.get('spread_bps_median')} "
        f"regime_gap_h_med={regime.get('flip_gap_hours_median')} "
        f"pct_1m_lt_fee={fee_moves.get('pct_1m_moves_smaller_than_fee_rt')}",
        flush=True,
    )
    print(f"Wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
