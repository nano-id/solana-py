"""CLI for the data layer."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from pairs_trading.data.config import load_config
from pairs_trading.data.pipeline import DataPipeline


def _extract_global_flags(argv: list[str]) -> tuple[list[str], Path | None, bool]:
    """Pull --config / -v from anywhere so order vs subcommand does not matter."""
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
        prog="pairs-data",
        description="Fetch and store crypto OHLCV + universe snapshots (Phase 1 data layer).",
    )
    # Globals are stripped by _extract_global_flags before parse_args; kept here
    # only so -h documents them.
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config (default: config/default.yaml). Position-independent.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Debug logging. Position-independent.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_universe = sub.add_parser("universe", help="Snapshot top-N universe by 24h quote volume")
    p_universe.add_argument(
        "--print",
        dest="print_json",
        action="store_true",
        help="Print snapshot JSON to stdout",
    )

    p_sync = sub.add_parser("sync", help="Snapshot universe (optional) and sync OHLCV parquet files")
    p_sync.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Explicit CCXT symbols (e.g. BTC/USDT ETH/USDT). Default: current universe.",
    )
    p_sync.add_argument(
        "--no-universe",
        action="store_true",
        help="Do not refresh universe snapshot (requires --symbols)",
    )
    p_sync.add_argument(
        "--full",
        action="store_true",
        help="Ignore local cache and re-fetch full lookback window",
    )

    p_show = sub.add_parser("show", help="Show stored OHLCV summary for a symbol")
    p_show.add_argument("symbol", help="CCXT symbol, e.g. BTC/USDT")
    p_show.add_argument("--tail", type=int, default=5, help="Print last N rows")

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw = list(sys.argv[1:] if argv is None else argv)
    cleaned, config, verbose = _extract_global_flags(raw)
    parser = build_parser()
    args = parser.parse_args(cleaned)
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

    config = load_config(args.config)
    pipeline = DataPipeline(config)

    if args.command == "universe":
        snapshot, path = pipeline.snapshot_universe()
        print(f"Wrote {path} ({len(snapshot.members)} symbols) via {pipeline.exchange_id}")
        if args.print_json:
            print(
                json.dumps(
                    {
                        "as_of": snapshot.as_of,
                        "exchange_id": snapshot.exchange_id,
                        "symbols": [
                            {
                                "rank": m.rank,
                                "symbol": m.symbol,
                                "quote_volume_24h": m.quote_volume_24h,
                            }
                            for m in snapshot.members
                        ],
                    },
                    indent=2,
                )
            )
        return 0

    if args.command == "sync":
        if args.no_universe and not args.symbols:
            # Re-build parser only for error formatting.
            build_parser().error("--no-universe requires --symbols")
        result = pipeline.run(
            symbols=args.symbols,
            refresh_universe=not args.no_universe,
            force_full=args.full,
        )
        print(f"Exchange: {result.exchange_id}")
        if result.universe_path:
            print(f"Universe: {result.universe_path}")
        print(f"Synced {len(result.synced)} symbols:")
        for item in result.synced:
            print(f"  {item.symbol}: +{item.bars_written} bars -> {item.path}")
        return 0 if result.synced else 1

    if args.command == "show":
        df = pipeline.store.load(args.symbol)
        if df.empty:
            print(f"No data for {args.symbol} on {pipeline.exchange_id}")
            return 1
        print(f"[{pipeline.exchange_id}] {args.symbol}: {len(df)} bars")
        print(f"from {df['datetime_utc'].iloc[0]} to {df['datetime_utc'].iloc[-1]}")
        print(df.tail(args.tail).to_string(index=False))
        return 0

    build_parser().error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
