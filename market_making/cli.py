"""CLI: spot-mm explain-gate | quote | simulate."""

from __future__ import annotations

import argparse
import json
import logging
import sys

import numpy as np

from market_making.config import load_mm_config
from market_making.engine import QuoteEngine
from market_making.fees import is_positive_half_spread, round_trip_net_edge_bps
from market_making.inventory import InventoryState
from market_making.simulator import run_month_sim, run_paper_sim, save_sim_artifacts
from market_making.portfolio import run_portfolio_month, save_portfolio_artifacts
from market_making.realtime import run_live_paper_session, run_realtime_paper, save_realtime_artifacts
from market_making.feed import BookSnapshot
from market_making.hist_bbo import run_month_events_paper, save_month_events_artifacts
from market_making.calibrate_hourly import (
    calibrate_hourly,
    fifo_hold_hours_from_fills,
    save_hourly_calib,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="spot-mm",
        description="Spot-only market making (fee=0.07% default; positivity-gated)",
    )
    p.add_argument("--config", default="config/market_making.yaml")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("explain-gate", help="Show fee / half-spread positivity math")
    g.add_argument("--half-spread-bps", type=float, default=None)

    q = sub.add_parser("quote", help="One-shot quote at a mid")
    q.add_argument("--mid", type=float, required=True)
    q.add_argument("--inventory", type=float, default=0.0)

    s = sub.add_parser("simulate", help="Paper GBM simulation with touch fills")
    s.add_argument("--n-bars", type=int, default=None)
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--no-persist", action="store_true")

    m = sub.add_parser(
        "simulate-month",
        help="Historical OKX spot MM over a calendar month (default: January)",
    )
    m.add_argument("--year", type=int, default=2026)
    m.add_argument("--month", type=int, default=1)
    m.add_argument("--symbol", default="BTC/USDT")
    m.add_argument("--timeframe", default="1h")
    m.add_argument("--seed", type=int, default=42)
    m.add_argument("--no-persist", action="store_true")

    pf = sub.add_parser(
        "simulate-portfolio",
        help="Single-wallet MM over all config symbols (lock-until-flat; best signal wins)",
    )
    pf.add_argument("--year", type=int, default=2024)
    pf.add_argument("--month", type=int, default=2)
    pf.add_argument("--timeframe", default="1h")
    pf.add_argument("--seed", type=int, default=42)
    pf.add_argument("--equity", type=float, default=None, help="Initial USDT wallet (default: config)")
    pf.add_argument("--no-persist", action="store_true")
    lp = sub.add_parser(
        "live-paper",
        help="Event-driven paper MM on live BBO (NOT candles); hold time in seconds",
    )
    lp.add_argument("--duration-sec", type=float, default=90.0)
    lp.add_argument("--poll-sec", type=float, default=None)
    lp.add_argument("--equity", type=float, default=1000.0)
    lp.add_argument("--seed", type=int, default=42)
    lp.add_argument("--no-persist", action="store_true")
    lp.add_argument("--synthetic", action="store_true", help="Synthetic BBO (offline)")

    ch = sub.add_parser(
        "calibrate-hourly",
        help="Re-optimize AS params on 1h candles (IS→OOS); update-friendly JSON",
    )
    ch.add_argument("--timeframe", default="1h")
    ch.add_argument("--seed", type=int, default=42)
    ch.add_argument("--equity", type=float, default=1000.0)
    ch.add_argument("--no-persist", action="store_true")
    ch.add_argument(
        "--write-config",
        action="store_true",
        help="Write selected params into --config YAML market_making block",
    )

    me = sub.add_parser(
        "simulate-month-events",
        help="Event-driven paper MM on OHLC→synth BBO (NOT candle decisions)",
    )
    me.add_argument("--year", type=int, default=2025)
    me.add_argument("--month", type=int, default=3)
    me.add_argument("--timeframe", default="1h")
    me.add_argument("--steps-per-bar", type=int, default=60)
    me.add_argument("--seed", type=int, default=42)
    me.add_argument("--equity", type=float, default=None)
    me.add_argument(
        "--fill-prob",
        type=float,
        default=None,
        help="Override sim.fill_prob (e.g. 0.25 conservative)",
    )
    me.add_argument(
        "--adverse-frac",
        type=float,
        default=None,
        help="Override sim.adverse_frac (e.g. 0.55 conservative)",
    )
    me.add_argument("--no-persist", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = load_mm_config(args.config)

    if args.command == "explain-gate":
        half = (
            args.half_spread_bps
            if args.half_spread_bps is not None
            else cfg.required_half_spread_bps
        )
        print(
            json.dumps(
                {
                    "commission_pct": cfg.commission_pct,
                    "commission_bps": cfg.commission_bps,
                    "edge_buffer_bps": cfg.edge_buffer_bps,
                    "min_half_spread_bps": cfg.min_half_spread_bps,
                    "required_half_spread_bps": cfg.required_half_spread_bps,
                    "example_half_spread_bps": half,
                    "round_trip_fee_bps": 2.0 * cfg.commission_bps,
                    "round_trip_net_edge_bps": round_trip_net_edge_bps(half, cfg),
                    "positive": is_positive_half_spread(half, cfg),
                    "interpretation_allowed": False,
                    "note": (
                        "Matched bid+ask earns ~2*half_spread minus 2*fee bps "
                        "before adverse selection. Gate refuses tighter quotes."
                    ),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "quote":
        inv = InventoryState(
            base=args.inventory,
            quote=cfg.max_inventory * args.mid * 2.0,
            target=cfg.target_inventory,
            max_abs=cfg.max_inventory,
            halt_frac=cfg.inventory_halt_frac,
        )
        eng = QuoteEngine(cfg)
        q = eng.quote(args.mid, inv, mid_history=np.asarray([args.mid]))
        print(
            json.dumps(
                {
                    "mid": q.mid,
                    "bid": q.bid,
                    "ask": q.ask,
                    "bid_size": q.bid_size,
                    "ask_size": q.ask_size,
                    "half_spread_bps": q.half_spread_bps,
                    "net_edge_bps_if_matched": q.net_edge_bps_if_matched,
                    "quoting": q.quoting,
                    "reason": q.reason,
                    "commission_pct": cfg.commission_pct,
                    "interpretation_allowed": False,
                },
                indent=2,
            )
        )
        return 0 if q.quoting else 2

    if args.command == "simulate":
        if args.n_bars is not None:
            cfg.sim.n_bars = args.n_bars
        if args.seed is not None:
            cfg.sim.seed = args.seed
        result, frame = run_paper_sim(cfg)
        paths: dict[str, str] = {}
        if not args.no_persist:
            paths = save_sim_artifacts(
                result,
                frame,
                root=cfg.storage_root,
                exchange_id=cfg.exchange_id,
            )
        print(json.dumps({**result.__dict__, "artifacts": paths}, indent=2))
        if result.n_buys + result.n_sells == 0:
            return 3
        return 0

    if args.command == "simulate-month":
        result, frame = run_month_sim(
            cfg,
            year=args.year,
            month=args.month,
            symbol=args.symbol,
            timeframe=args.timeframe,
            seed=args.seed,
        )
        paths: dict[str, str] = {}
        if not args.no_persist:
            tag = f"{args.symbol.replace('/', '_')}_{args.year:04d}{args.month:02d}"
            paths = save_sim_artifacts(
                result,
                frame,
                root=cfg.storage_root,
                exchange_id=cfg.exchange_id,
                basename=f"hist_{tag}",
            )
            # Also refresh "latest" pointer for convenience.
            save_sim_artifacts(
                result,
                frame,
                root=cfg.storage_root,
                exchange_id=cfg.exchange_id,
                basename="paper_sim_latest",
            )
        print(
            json.dumps(
                {
                    "year": args.year,
                    "month": args.month,
                    "symbol": args.symbol,
                    "timeframe": args.timeframe,
                    **result.__dict__,
                    "artifacts": paths,
                },
                indent=2,
                default=str,
            )
        )
        if result.n_buys + result.n_sells == 0:
            return 3
        return 0

    if args.command == "simulate-portfolio":
        result, frame = run_portfolio_month(
            cfg,
            year=args.year,
            month=args.month,
            timeframe=args.timeframe,
            seed=args.seed,
            initial_equity=args.equity,
        )
        paths: dict[str, str] = {}
        if not args.no_persist:
            tag = f"portfolio_{args.year:04d}{args.month:02d}"
            paths = save_portfolio_artifacts(
                result,
                frame,
                root=cfg.storage_root,
                exchange_id=cfg.exchange_id,
                basename=tag,
            )
        print(
            json.dumps(
                {
                    "year": args.year,
                    "month": args.month,
                    "timeframe": args.timeframe,
                    **result.__dict__,
                    "artifacts": paths,
                },
                indent=2,
                default=str,
            )
        )
        return 0 if result.n_entries > 0 else 3

    if args.command == "live-paper":
        if args.synthetic:
            import time as _time
            syms = list(cfg.symbols)[:5]
            mids = {s: 100.0 + i for i, s in enumerate(syms)}
            t0 = int(_time.time() * 1000)
            events = []
            n_ticks = max(30, int(args.duration_sec))
            for i in range(n_ticks):
                for s in syms:
                    mids[s] *= 1.0 + 0.00015 * ((i + hash(s)) % 5 - 2)
                    m = mids[s]
                    sp = m * 0.00025
                    events.append(
                        BookSnapshot(
                            symbol=s,
                            ts_ms=t0 + i * 1000,
                            bid=m - sp / 2,
                            ask=m + sp / 2,
                            bid_size=1.0,
                            ask_size=1.0,
                            mid=m,
                            spread_bps=2.5,
                            exchange_id="synthetic",
                        )
                    )
            result, rows = run_realtime_paper(
                cfg, events, seed=args.seed, initial_equity=args.equity,
                poll_interval_sec=args.poll_sec or 1.0,
            )
        else:
            result, rows = run_live_paper_session(
                cfg,
                duration_sec=args.duration_sec,
                poll_interval_sec=args.poll_sec,
                seed=args.seed,
                initial_equity=args.equity,
            )
        paths = {}
        if not args.no_persist:
            paths = save_realtime_artifacts(
                result, rows, root=cfg.storage_root, exchange_id=cfg.exchange_id,
                basename="realtime_paper_latest",
            )
        print(json.dumps({**result.__dict__, "artifacts": paths}, indent=2, default=str))
        return 0 if result.n_entries > 0 else 3

    if args.command == "calibrate-hourly":
        from market_making.calibrate_hourly import write_params_to_yaml

        result = calibrate_hourly(
            cfg,
            timeframe=args.timeframe,
            seed=args.seed,
            portfolio_equity=args.equity,
        )
        paths: dict = {}
        if not args.no_persist:
            paths = save_hourly_calib(
                result,
                root=cfg.storage_root,
                exchange_id=cfg.exchange_id,
                basename="hourly_calib_1h_202503",
            )
        if args.write_config:
            write_params_to_yaml(args.config, result.selected_params)
            paths["config"] = args.config
        print(
            json.dumps(
                {
                    **{k: getattr(result, k) for k in result.__dataclass_fields__},
                    "artifacts": paths,
                },
                indent=2,
                default=str,
            )
        )
        return 0

    if args.command == "simulate-month-events":
        result, rows, meta = run_month_events_paper(
            cfg,
            year=args.year,
            month=args.month,
            timeframe=args.timeframe,
            steps_per_bar=args.steps_per_bar,
            seed=args.seed,
            initial_equity=args.equity,
            fill_prob=args.fill_prob,
            adverse_frac=args.adverse_frac,
        )
        paths: dict = {}
        if not args.no_persist:
            tag = f"events_{args.year:04d}{args.month:02d}_{args.timeframe}"
            if args.fill_prob is not None:
                tag += f"_fp{args.fill_prob:.2f}"
            paths = save_month_events_artifacts(
                result,
                rows,
                meta,
                root=cfg.storage_root,
                exchange_id=cfg.exchange_id,
                basename=tag,
            )
        out = {
            **meta,
            "equity_start": result.start_equity,
            "equity_final": result.final_equity,
            "total_return_pct": result.total_return_pct,
            "n_entries": result.n_entries,
            "n_round_trips": result.n_round_trips,
            "n_skipped_busy": result.n_skipped_busy,
            "hold_seconds_mean": result.hold_seconds_mean,
            "hold_seconds_median": result.hold_seconds_median,
            "hold_minutes_mean": (
                None
                if result.hold_seconds_mean is None
                else result.hold_seconds_mean / 60.0
            ),
            "hold_minutes_median": (
                None
                if result.hold_seconds_median is None
                else result.hold_seconds_median / 60.0
            ),
            "fees_paid": result.fees_paid,
            "interpretation_allowed": False,
            "artifacts": paths,
        }
        print(json.dumps(out, indent=2, default=str))
        return 0 if result.n_entries > 0 else 3

    return 1


if __name__ == "__main__":
    sys.exit(main())
