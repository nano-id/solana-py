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
    m.add_argument("--timeframe", default="4h")
    m.add_argument("--seed", type=int, default=42)
    m.add_argument("--no-persist", action="store_true")
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

    return 1


if __name__ == "__main__":
    sys.exit(main())
