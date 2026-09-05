"""CLI for funding_arb — collect, screen (floor-aware), paper entry."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from funding_arb.config import load_funding_arb_config
from funding_arb.entry import (
    paper_dual_leg_entry,
    select_entry_candidate,
)
from funding_arb.loop import run_collect_cycle, run_collect_loop
from funding_arb.screen import (
    load_history,
    save_screen_batch,
    screen_opportunities,
    screen_summary,
)
from funding_arb.storage import snapshot_summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="funding-arb",
        description="Funding-rate arbitrage engine (collect + screen + paper entry)",
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
        help="Rank annualized funding; prefer non-floor premium; depth filter",
    )
    s.add_argument("--symbols", nargs="+", default=None)
    s.add_argument("--min-depth", type=float, default=None)
    s.add_argument(
        "--from-latest",
        action="store_true",
        help="Screen from saved latest snapshot (no live fetch)",
    )
    s.add_argument("--no-persist", action="store_true")

    e = sub.add_parser(
        "enter",
        help="Paper delta-neutral entry from top screen candidate (step 3)",
    )
    e.add_argument("--symbols", nargs="+", default=None)
    e.add_argument("--from-latest", action="store_true", default=True)
    e.add_argument("--live-screen", action="store_true", help="Refresh screen live first")
    e.add_argument(
        "--allow-floor",
        action="store_true",
        help="Allow interest-floor names (default: prefer real premium)",
    )
    e.add_argument(
        "--simulate-perp-miss",
        action="store_true",
        help="Paper-only: force perp timeout to exercise unpaired-leg flatten",
    )
    e.add_argument(
        "--simulate-spot-miss",
        action="store_true",
        help="Paper-only: force spot timeout to exercise unpaired-leg flatten",
    )
    return p


def _load_latest_snapshot(root: Path, exchange_ids: list[str]) -> tuple[pd.DataFrame, str]:
    for eid in exchange_ids:
        path = root / eid / "funding_arb" / "snapshots" / "latest.parquet"
        if path.is_file():
            return pd.read_parquet(path), eid
    raise FileNotFoundError("No latest snapshot. Run: funding-arb collect --once")


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
            "premium_rate",
            "sett_funding_rate",
            "at_interest_floor",
            "floor_persistence",
            "depth_usdt",
            "passes_depth",
            "side_hint",
        ]
        if c in frame.columns
    ]
    print(frame[cols].to_string(index=False, float_format=lambda x: f"{x:.6g}"))


def _build_screen(config, frame: pd.DataFrame, exchange_id: str, min_depth: float) -> pd.DataFrame:
    history = load_history(config.storage.root, exchange_id)
    return screen_opportunities(
        frame,
        settlements_per_day=config.collector.settlements_per_day,
        min_depth_usdt=min_depth,
        rate_col=config.screen.rate_col,
        history=history,
        floor_lookback=config.screen.floor_lookback,
        prefer_non_floor=config.screen.prefer_non_floor,
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
            candidates = [config.exchange.id, *list(config.exchange.fallbacks), "okx"]
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

        screen = _build_screen(config, frame, exchange_id, min_depth)
        print(f"=== SCREEN (min_depth_usdt={min_depth:g}, prefer_non_floor={config.screen.prefer_non_floor}) ===")
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

    if args.command == "enter":
        min_depth = float(config.screen.min_depth_usdt)
        if args.live_screen:
            frame, connected, _paths = run_collect_cycle(
                config, symbols=args.symbols, persist=True
            )
            exchange_id = connected.exchange_id
        else:
            candidates = [config.exchange.id, *list(config.exchange.fallbacks), "okx"]
            frame, exchange_id = _load_latest_snapshot(
                Path(config.storage.root), candidates
            )
            print(f"enter from_latest exchange={exchange_id}")

        if args.symbols:
            frame = frame[frame["symbol"].isin(args.symbols)].reset_index(drop=True)

        screen = _build_screen(config, frame, exchange_id, min_depth)
        cand = select_entry_candidate(
            screen,
            require_non_floor=not args.allow_floor and config.entry.require_non_floor,
            require_depth=True,
        )
        if cand is None:
            print("No entry candidate (depth/floor filters).")
            return 2

        spot_ratio = 0.0 if args.simulate_spot_miss else 1.0
        perp_ratio = 0.0 if args.simulate_perp_miss else 1.0
        result = paper_dual_leg_entry(
            str(cand["symbol"]),
            depth_usdt=float(cand.get("depth_usdt") or 0.0),
            side_hint=str(cand.get("side_hint") or "flat"),
            depth_fraction=config.entry.depth_fraction,
            max_notional_usdt=config.entry.max_notional_usdt,
            min_notional_usdt=config.entry.min_notional_usdt,
            spot_fill_ratio=spot_ratio,
            perp_fill_ratio=perp_ratio,
            flatten_unpaired=config.entry.flatten_unpaired,
        )
        print("=== PAPER ENTRY ===")
        print(
            {
                "symbol": result.symbol,
                "side_hint": result.side_hint,
                "target_usdt": result.target_usdt,
                "state": result.state.value,
                "residual_usdt": result.residual_usdt,
                "at_interest_floor": cand.get("at_interest_floor"),
                "floor_persistence": cand.get("floor_persistence"),
                "premium_rate": cand.get("premium_rate"),
                "ann_funding": cand.get("ann_funding"),
                "actions": result.actions,
                "alarms": result.alarms,
                "spot": {
                    "filled": result.spot.filled_usdt,
                    "status": result.spot.status.value,
                },
                "perp": {
                    "filled": result.perp.filled_usdt,
                    "status": result.perp.status.value,
                },
            }
        )
        return 0 if result.state.value in {"balanced", "flattened", "aborted"} else 1

    build_parser().error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
