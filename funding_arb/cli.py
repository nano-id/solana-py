"""CLI for funding_arb — step 1 collect, step 2 screen."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from funding_arb.config import load_funding_arb_config
from funding_arb.loop import run_collect_cycle, run_collect_loop
from funding_arb.screen import save_screen_batch, screen_opportunities, screen_summary
from funding_arb.storage import snapshot_summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="funding-arb",
        description="Funding-rate arbitrage engine (collect + screen)",
    )
    p.add_argument("--config", default="config/funding_arb.yaml")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("collect", help="Fetch funding + prices + order-book depth")
    c.add_argument("--symbols", nargs="+", default=None)
    c.add_argument("--once", action="store_true", help="Single snapshot (default)")
    c.add_argument("--loop", action="store_true")
    c.add_argument("--max-cycles", type=int, default=None)
    c.add_argument("--no-persist", action="store_true")

    s = sub.add_parser(
        "screen",
        help="Rank annualized funding; filter by min depth (step 2)",
    )
    s.add_argument("--symbols", nargs="+", default=None)
    s.add_argument(
        "--min-depth",
        type=float,
        default=None,
        help="Override screen.min_depth_usdt",
    )
    s.add_argument(
        "--from-latest",
        action="store_true",
        help="Screen from saved latest snapshot (no live fetch)",
    )
    s.add_argument("--no-persist", action="store_true")
    return p


def _print_collect(frame: pd.DataFrame) -> None:
    cols = [
        c
        for c in [
            "symbol",
            "funding_rate",
            "funding_rate_raw",
            "sett_funding_rate",
            "at_interest_floor",
            "spot_last",
            "perp_last",
            "perp_contract_size",
            "depth_usdt",
        ]
        if c in frame.columns
    ]
    view = frame[cols].copy()
    if "funding_rate" in view.columns:
        view = view.sort_values("funding_rate", ascending=False, key=lambda s: s.abs())
    print(view.to_string(index=False, float_format=lambda x: f"{x:.8g}"))


def _print_screen(frame: pd.DataFrame) -> None:
    cols = [
        c
        for c in [
            "rank",
            "symbol",
            "funding_rate",
            "ann_funding",
            "sett_funding_rate",
            "ann_sett_funding",
            "at_interest_floor",
            "depth_usdt",
            "passes_depth",
            "side_hint",
        ]
        if c in frame.columns
    ]
    print(frame[cols].to_string(index=False, float_format=lambda x: f"{x:.6g}"))


def _load_latest_snapshot(root: Path, exchange_ids: list[str]) -> tuple[pd.DataFrame, str]:
    for eid in exchange_ids:
        path = root / eid / "funding_arb" / "snapshots" / "latest.parquet"
        if path.is_file():
            return pd.read_parquet(path), eid
    raise FileNotFoundError(
        "No latest snapshot found. Run: funding-arb collect --once"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_funding_arb_config(args.config)

    if args.command == "collect":
        persist = not args.no_persist
        if args.loop:
            n = run_collect_loop(
                config,
                symbols=args.symbols,
                max_cycles=args.max_cycles,
                persist=persist,
            )
            print(f"completed_cycles={n}")
            return 0
        frame, connected, paths = run_collect_cycle(
            config, symbols=args.symbols, persist=persist
        )
        print(f"exchange={connected.exchange_id}  {snapshot_summary(frame)}")
        _print_collect(frame)
        if paths:
            print("Wrote:", ", ".join(str(p) for p in paths[:8]))
        return 0 if not frame.empty else 1

    if args.command == "screen":
        min_depth = (
            float(args.min_depth)
            if args.min_depth is not None
            else float(config.screen.min_depth_usdt)
        )
        if args.from_latest:
            candidates = [
                config.exchange.id,
                *list(config.exchange.fallbacks),
                "okx",
            ]
            frame, exchange_id = _load_latest_snapshot(
                Path(config.storage.root), candidates
            )
            print(f"screen from_latest exchange={exchange_id}")
        else:
            frame, connected, _paths = run_collect_cycle(
                config, symbols=args.symbols, persist=True
            )
            exchange_id = connected.exchange_id

        if args.symbols:
            frame = frame[frame["symbol"].isin(args.symbols)].reset_index(drop=True)

        screen = screen_opportunities(
            frame,
            settlements_per_day=config.collector.settlements_per_day,
            min_depth_usdt=min_depth,
            rate_col=config.screen.rate_col,
        )
        print(f"=== SCREEN (min_depth_usdt={min_depth:g}) ===")
        print(screen_summary(screen))
        _print_screen(screen)
        if not args.no_persist:
            paths = save_screen_batch(
                screen,
                root=config.storage.root,
                exchange_id=exchange_id,
                mode="both",
                min_depth_usdt=min_depth,
            )
            if paths:
                print("Wrote:", ", ".join(str(p) for p in paths))
        eligible = int(screen["passes_depth"].sum()) if not screen.empty else 0
        return 0 if eligible > 0 else 2

    build_parser().error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
