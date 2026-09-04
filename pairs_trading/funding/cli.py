"""CLI for Module 1 — perpetual funding-rate data + screens."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pairs_trading.funding.config import load_funding_config
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
        description="Module 1: perpetual funding-rate sync and harvest screens.",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sync = sub.add_parser("sync", help="Fetch funding-rate history → parquet")
    p_sync.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="CCXT swap symbols, e.g. BTC/USDT:USDT",
    )
    p_sync.add_argument("--full", action="store_true", help="Ignore local cache")

    p_show = sub.add_parser("show", help="Show stored funding summary + tail")
    p_show.add_argument("symbol", help="e.g. BTC/USDT:USDT")
    p_show.add_argument("--tail", type=int, default=5)

    p_screen = sub.add_parser("screen", help="Rank stored symbols by |annualized mean funding|")
    p_screen.add_argument("--symbols", nargs="+", default=None)

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
        # Prefer whatever was last written — try okx if default file missing
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
        # Screen across whichever venue has local funding parquet
        tried = []
        df = None
        for eid in [config.exchange.id, *config.exchange.fallbacks]:
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
            f"{config.funding.settlements_per_day} × 365 (simple). "
            "Not a live entry rule — capacity/basis/fees come next."
        )
        return 0

    build_parser().error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
