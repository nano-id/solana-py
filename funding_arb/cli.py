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
from funding_arb.journal import JournalCosts, load_journal, record_cycle
from funding_arb.report import build_observation_report, save_observation_report
from funding_arb.control import run_control_cycle
from funding_arb.position import PaperPosition


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

    r = sub.add_parser(
        "report",
        help="Observation report: floor_persistence + journal gross/net (step 6)",
    )
    r.add_argument("--exchange-id", default=None)
    r.add_argument("--min-days", type=float, default=None)
    r.add_argument("--persist", action="store_true", default=True)

    y = sub.add_parser(
        "cycle",
        help="Paper control cycle with journal logging (exit→rotate/enter)",
    )
    y.add_argument("--from-latest", action="store_true", default=True)
    y.add_argument("--position-json", default=None, help="Optional open PaperPosition JSON path")
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


    if args.command == "report":
        candidates = [config.exchange.id, *list(config.exchange.fallbacks), "okx"]
        exchange_id = args.exchange_id
        if exchange_id is None:
            # Prefer exchange that already has history.
            from funding_arb.storage import snapshots_dir
            exchange_id = next(
                (
                    eid
                    for eid in candidates
                    if (snapshots_dir(config.storage.root, eid) / "history.parquet").is_file()
                ),
                candidates[-1],
            )
        min_days = (
            float(args.min_days)
            if args.min_days is not None
            else float(config.observation.min_days)
        )
        report = build_observation_report(
            root=config.storage.root,
            exchange_id=exchange_id,
            floor_lookback=config.observation.floor_lookback,
            min_days=min_days,
            min_settlements=config.observation.min_settlements,
            settlements_per_day=config.collector.settlements_per_day,
        )
        print("=== OBSERVATION REPORT ===")
        import json as _json
        print(_json.dumps(report, indent=2))
        if args.persist:
            path = save_observation_report(
                report, root=config.storage.root, exchange_id=exchange_id
            )
            print("Wrote:", path)
        return 0 if report["readiness"]["ready"] else 3

    if args.command == "cycle":
        import json as _json
        candidates = [config.exchange.id, *list(config.exchange.fallbacks), "okx"]
        frame, exchange_id = _load_latest_snapshot(Path(config.storage.root), candidates)
        min_depth = float(config.screen.min_depth_usdt)
        screen = _build_screen(config, frame, exchange_id, min_depth)
        pos_path = Path(config.storage.root) / exchange_id / "funding_arb" / "paper_position.json"
        position = None
        position_src = Path(args.position_json) if args.position_json else pos_path
        if position_src.is_file():
            raw = _json.loads(position_src.read_text(encoding="utf-8"))
            position = PaperPosition.from_dict(raw)
            print(f"cycle loaded_position={position_src} symbol={position.symbol}")
        result = run_control_cycle(
            screen,
            position,
            exchange_id=exchange_id,
            min_edge_multiple=config.rotation.min_edge_multiple,
            rotation_cost=config.rotation.rotation_cost,
            require_non_floor=config.rotation.require_non_floor,
            exit_on_floor=config.exit.exit_on_floor,
            exit_on_adverse_funding=config.exit.exit_on_adverse_funding,
            depth_fraction=config.entry.depth_fraction,
            max_notional_usdt=config.entry.max_notional_usdt,
            min_notional_usdt=config.entry.min_notional_usdt,
            flatten_unpaired=config.entry.flatten_unpaired,
        )
        print("=== CONTROL CYCLE ===")
        print(_json.dumps(result.to_dict(), indent=2, default=str))
        if config.journal.enabled:
            paths = record_cycle(
                result,
                root=config.storage.root,
                exchange_id=exchange_id,
                costs=JournalCosts(
                    rotation_cost=config.journal.rotation_cost,
                    slippage_cost=config.journal.slippage_cost,
                ),
                settlements_per_day=config.collector.settlements_per_day,
            )
            if paths:
                print("Journal:", ", ".join(str(p) for p in paths[:3]))
        pos_path.parent.mkdir(parents=True, exist_ok=True)
        if result.position_after is not None:
            pos_path.write_text(
                _json.dumps(result.position_after.to_dict(), indent=2) + "\n",
                encoding="utf-8",
            )
            print("Position:", pos_path)
        elif pos_path.is_file() and result.action.value in {"exit", "flatten_alarm"}:
            pos_path.unlink()
            print("Position cleared")
        return 0


    build_parser().error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
