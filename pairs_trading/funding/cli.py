"""CLI for Module 1 — perpetual funding sync, screens, harvest, paper."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pairs_trading.data.exchange import connect_with_fallback
from pairs_trading.funding.config import load_funding_config
from pairs_trading.funding.harvest import build_harvest_table, run_paper_backtest
from pairs_trading.funding.persist import save_funding_screen
from pairs_trading.funding.pipeline import FundingPipeline
from pairs_trading.funding.screen import screen_funding_store
from pairs_trading.funding.storage import FundingStore


def _extract_global_flags(argv: list[str]) -> tuple[list[str], Path | None, bool]:
    config: Path | None = None
    verbose = False
    cleaned: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--config" and i + 1 < len(argv):
            config = Path(argv[i + 1])
            i += 2
            continue
        if arg.startswith("--config="):
            config = Path(arg.split("=", 1)[1])
            i += 1
            continue
        if arg in ("-v", "--verbose"):
            verbose = True
            i += 1
            continue
        cleaned.append(arg)
        i += 1
    return cleaned, config, verbose


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pairs-funding",
        description="Module 1: perpetual funding-rate harvest pipeline.",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sync = sub.add_parser("sync", help="Fetch funding-rate history → parquet")
    p_sync.add_argument("--symbols", nargs="+", default=None)
    p_sync.add_argument("--full", action="store_true", help="Ignore local cache")

    p_show = sub.add_parser("show", help="Show stored funding summary + tail")
    p_show.add_argument("symbol")
    p_show.add_argument("--tail", type=int, default=5)

    p_screen = sub.add_parser("screen", help="Rank by |ann_mean| funding")
    p_screen.add_argument("--symbols", nargs="+", default=None)

    p_harvest = sub.add_parser(
        "harvest",
        help="Net-carry screen: fees + persistence + basis + capacity + size",
    )
    p_harvest.add_argument("--symbols", nargs="+", default=None)
    p_harvest.add_argument("--no-live", action="store_true", help="Skip live basis/volume")
    p_harvest.add_argument("--persist", action="store_true", help="Write screen artifacts")

    p_paper = sub.add_parser("paper", help="Paper backtest harvest on stored history")
    p_paper.add_argument("--symbols", nargs="+", default=None)
    p_paper.add_argument("--no-live", action="store_true")
    p_paper.add_argument("--persist", action="store_true")

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw = list(sys.argv[1:] if argv is None else argv)
    cleaned, config, verbose = _extract_global_flags(raw)
    args = build_parser().parse_args(cleaned)
    if config is not None:
        args.config = config
    if verbose:
        args.verbose = True
    return args


def _resolve_store_exchange_id(config) -> str:
    for eid in config.exchange.candidate_ids():
        store = FundingStore(config.storage.root, eid)
        if store.list_symbols():
            return eid
    return config.exchange.id


def _persist_mode(config) -> str:
    mode = config.funding.persist_mode
    return mode if mode in ("dated", "latest", "both") else "both"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_funding_config(args.config)

    if args.command == "sync":
        pipe = FundingPipeline(config)
        results = pipe.sync_many(args.symbols, force_full=args.full)
        print(f"Exchange: {pipe.exchange_id} (swap)")
        if not results:
            print("No symbols synced.")
            return 1
        for r in results:
            print(f"  {r.symbol}: +{r.rows_written} rows → {r.path}")
        return 0

    if args.command == "show":
        eid = config.exchange.id
        store = FundingStore(config.storage.root, eid)
        if not store.exists(args.symbol):
            for cand in config.exchange.candidate_ids():
                alt = FundingStore(config.storage.root, cand)
                if alt.exists(args.symbol):
                    store = alt
                    eid = cand
                    break
        df = store.load(args.symbol)
        print(f"exchange={eid} symbol={args.symbol} n={len(df)}")
        if df.empty:
            return 1
        print(df.tail(args.tail).to_string(index=False))
        return 0

    if args.command == "screen":
        tried: list[str] = []
        df = None
        for eid in config.exchange.candidate_ids():
            tried.append(eid)
            cand = screen_funding_store(config, symbols=args.symbols, exchange_id=eid)
            if not cand.empty:
                df = cand
                break
        if df is None or df.empty:
            print(f"No funding parquet under {tried}. Run: pairs-funding sync")
            return 1
        cols = [
            c
            for c in [
                "rank",
                "symbol",
                "n_obs",
                "last_rate",
                "mean_rate",
                "ann_mean",
                "ann_recent",
                "pct_positive",
                "above_noise",
            ]
            if c in df.columns
        ]
        print("=== FUNDING HARVEST SCREEN (|ann_mean| desc) ===")
        print(df[cols].to_string(index=False))
        print(
            "\nNote: ann_* = mean_settlement_rate × "
            f"{config.funding.settlements_per_day} × 365 (simple)."
        )
        return 0

    if args.command in ("harvest", "paper"):
        eid = _resolve_store_exchange_id(config)
        exchange = None
        if not args.no_live:
            connected = connect_with_fallback(config.exchange)
            exchange = connected.exchange
            eid = connected.exchange_id
            config.exchange.id = eid

        table = build_harvest_table(
            config, exchange=exchange, exchange_id=eid, symbols=args.symbols
        )
        if table.empty:
            print("No harvest rows. Run: pairs-funding sync")
            return 1

        if args.command == "harvest":
            cols = [
                c
                for c in [
                    "rank",
                    "symbol",
                    "side",
                    "ann_mean",
                    "gross_ann",
                    "fee_drag_ann",
                    "basis_drag_ann",
                    "net_ann",
                    "harvest_efficiency",
                    "basis_bps",
                    "quote_volume_24h",
                    "sized_notional",
                    "tradable",
                    "positive_net",
                ]
                if c in table.columns
            ]
            print("=== NET CARRY HARVEST (net_ann desc) ===")
            print(table[cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
            n_pos = int(table["positive_net"].sum())
            n_tradable = int(table["tradable"].sum())
            print(f"\npositive_net={n_pos} tradable={n_tradable} / {len(table)}")
            if args.persist:
                paths = save_funding_screen(
                    table,
                    root=config.storage.root,
                    exchange_id=eid,
                    name="harvest",
                    mode=_persist_mode(config),  # type: ignore[arg-type]
                    meta={"positive_net": n_pos, "tradable": n_tradable},
                )
                print("Wrote:", ", ".join(str(p) for p in paths))
            return 0 if n_pos > 0 else 2

        _, paper_df = run_paper_backtest(
            config, exchange=exchange, exchange_id=eid, harvest_df=table
        )
        if paper_df.empty:
            print("No paper results. Run: pairs-funding sync")
            return 1
        paper_df = paper_df.sort_values("net_pnl", ascending=False).reset_index(drop=True)
        cols = [
            c
            for c in [
                "symbol",
                "side",
                "n_settlements",
                "notional_usdt",
                "funding_pnl",
                "fee_pnl",
                "basis_pnl",
                "net_pnl",
                "net_return",
                "ann_return",
                "hold_days",
                "positive",
            ]
            if c in paper_df.columns
        ]
        print("=== PAPER FUNDING HARVEST ===")
        print(paper_df[cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
        n_pos = int(paper_df["positive"].sum())
        print(f"\npositive_pnl={n_pos} / {len(paper_df)}")
        if args.persist:
            paths = save_funding_screen(
                paper_df,
                root=config.storage.root,
                exchange_id=eid,
                name="paper",
                mode=_persist_mode(config),  # type: ignore[arg-type]
                meta={"positive": n_pos},
            )
            print("Wrote:", ", ".join(str(p) for p in paths))
        return 0 if n_pos > 0 else 2

    build_parser().error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
