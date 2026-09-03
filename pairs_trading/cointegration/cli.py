"""CLI for cointegration scanning."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pairs_trading.cointegration.engine import evaluate_pair
from pairs_trading.cointegration.scanner import scan_pairs
from pairs_trading.data.config import load_config
from pairs_trading.data.pipeline import DataPipeline
from pairs_trading.data.storage import ParquetStore


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
        prog="pairs-coint",
        description="Engle-Granger cointegration scan (Phase 2).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config. Position-independent.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="Scan universe pairs")
    p_scan.add_argument("--symbols", nargs="+", default=None, help="Override universe")
    p_scan.add_argument(
        "--as-of",
        dest="universe_as_of",
        default=None,
        help="Universe snapshot date YYYY-MM-DD or 'auto'",
    )
    p_scan.add_argument("--no-persist", action="store_true", help="Do not write results")
    p_scan.add_argument("--exchange", default=None, help="Override exchange id for local parquet")

    p_pair = sub.add_parser("pair", help="Test a single pair (always reports)")
    p_pair.add_argument("y", help="Dependent symbol, e.g. ETH/USDT")
    p_pair.add_argument("x", help="Independent symbol, e.g. BTC/USDT")
    p_pair.add_argument("--sync", action="store_true", help="Sync OHLCV before testing")
    p_pair.add_argument("--exchange", default=None, help="Override exchange id for local data")

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
    config = load_config(args.config)

    if args.command == "scan":
        df, hits = scan_pairs(
            config,
            symbols=args.symbols,
            exchange_id=args.exchange,
            universe_as_of=args.universe_as_of,
            persist=not args.no_persist,
        )
        if df.empty:
            print("No cointegrated pairs found.")
            return 0
        cols = [
            "y_symbol",
            "x_symbol",
            "pvalue",
            "test_statistic",
            "hedge_ratio",
            "half_life",
            "n_obs",
        ]
        print(df[cols].to_string(index=False))
        print(f"\n{len(hits)} hits")
        return 0

    if args.command == "pair":
        if args.sync:
            pipe = DataPipeline(config)
            pipe.sync_symbol(args.y, force_full=True)
            pipe.sync_symbol(args.x, force_full=True)
            eid = pipe.exchange_id
        else:
            eid = args.exchange or config.exchange.id

        store = ParquetStore(config.storage.root, eid, config.ohlcv.timeframe)
        y_df = store.load(args.y)
        x_df = store.load(args.x)
        if y_df.empty or x_df.empty:
            print(f"Missing data for {args.y} or {args.x} under {store.base_dir}")
            return 1
        hit = evaluate_pair(
            args.y,
            args.x,
            y_df,
            x_df,
            min_obs=config.cointegration.min_obs,
            price_space=config.cointegration.price_space,
        )
        print(f"exchange={eid}")
        print(f"price_space={hit.price_space}")
        print(f"pair={hit.y_symbol} ~ {hit.x_symbol}")
        print(f"n_obs={hit.n_obs}")
        print(f"pvalue={hit.pvalue:.6f}")
        print(f"test_statistic={hit.test_statistic:.4f}")
        print(f"hedge_ratio_beta={hit.hedge_ratio:.6f}")
        print(f"intercept={hit.intercept:.6f}")
        print(f"half_life_bars={hit.half_life:.4f}")
        return 0

    build_parser().error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
