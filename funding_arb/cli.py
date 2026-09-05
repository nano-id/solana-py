"""CLI for funding_arb step 1 — live market snapshot collection."""

from __future__ import annotations

import argparse
import logging
import sys

from funding_arb.config import load_funding_arb_config
from funding_arb.loop import run_collect_cycle, run_collect_loop
from funding_arb.storage import snapshot_summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="funding-arb",
        description="Funding-rate arbitrage engine (step 1: live data collector)",
    )
    p.add_argument("--config", default="config/funding_arb.yaml")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("collect", help="Fetch funding + prices + order-book depth")
    c.add_argument("--symbols", nargs="+", default=None)
    c.add_argument(
        "--once",
        action="store_true",
        help="Single snapshot then exit (default if --loop not set)",
    )
    c.add_argument(
        "--loop",
        action="store_true",
        help="Repeat every collector.interval_seconds (default 900)",
    )
    c.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="With --loop, stop after N cycles (default: forever)",
    )
    c.add_argument("--no-persist", action="store_true")
    c.add_argument(
        "--print",
        dest="print_table",
        action="store_true",
        help="Print a compact table to stdout",
    )
    return p


def _print_frame(frame) -> None:
    cols = [
        c
        for c in [
            "symbol",
            "funding_rate",
            "spot_last",
            "perp_last",
            "basis_bps_spot",
            "depth_usdt",
            "perp_spread_bps",
            "quote_volume_24h",
        ]
        if c in frame.columns
    ]
    view = frame[cols].copy()
    if "funding_rate" in view.columns:
        view = view.sort_values("funding_rate", ascending=False, key=lambda s: s.abs())
    print(view.to_string(index=False, float_format=lambda x: f"{x:.6g}"))


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
                on_cycle=(
                    (lambda frame, _s: _print_frame(frame)) if args.print_table else None
                ),
            )
            print(f"completed_cycles={n}")
            return 0

        # Default: one-shot (also when --once is passed).
        frame, connected, paths = run_collect_cycle(
            config,
            symbols=args.symbols,
            persist=persist,
        )
        print(f"exchange={connected.exchange_id}  {snapshot_summary(frame)}")
        _print_frame(frame)
        if paths:
            print("Wrote:", ", ".join(str(p) for p in paths[:6]), end="")
            if len(paths) > 6:
                print(f", … (+{len(paths) - 6} more)")
            else:
                print()
        return 0 if not frame.empty else 1

    build_parser().error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
