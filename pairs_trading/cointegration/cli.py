"""CLI for cointegration scanning."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from pairs_trading.cointegration.engine import evaluate_pair
from pairs_trading.cointegration.hub import run_hub_placebo_tests, summarize_hub_from_scan
from pairs_trading.cointegration.persist import save_scan_results
from pairs_trading.cointegration.scanner import resolve_exchange_id, scan_pairs
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

    p_scan = sub.add_parser("scan", help="Scan universe pairs (with BH-FDR)")
    p_scan.add_argument("--symbols", nargs="+", default=None, help="Override universe")
    p_scan.add_argument(
        "--as-of",
        dest="universe_as_of",
        default=None,
        help="Universe snapshot date YYYY-MM-DD or 'auto'",
    )
    p_scan.add_argument("--no-persist", action="store_true", help="Do not write results")
    p_scan.add_argument("--exchange", default=None, help="Override exchange id for local parquet")
    p_scan.add_argument(
        "--preprocess",
        choices=["none", "btc_residual"],
        default=None,
        help="Override config preprocess (btc_residual = factor-neutral residuals)",
    )

    p_pair = sub.add_parser("pair", help="Test a single pair (always reports)")
    p_pair.add_argument("y", help="Dependent symbol, e.g. ETH/USDT")
    p_pair.add_argument("x", help="Independent symbol, e.g. BTC/USDT")
    p_pair.add_argument("--sync", action="store_true", help="Sync OHLCV before testing")
    p_pair.add_argument("--exchange", default=None, help="Override exchange id for local data")

    p_hub = sub.add_parser(
        "hub-test",
        help="Hub diagnostic: test a symbol vs placebo/unrelated coins",
    )
    p_hub.add_argument("--hub", default="BNB/USDT", help="Suspected hub symbol")
    p_hub.add_argument(
        "--placebos",
        nargs="+",
        default=None,
        help="Placebo symbols (default: gold/meme/unrelated L1s)",
    )
    p_hub.add_argument(
        "--sync-placebos",
        action="store_true",
        help="Sync hub + placebos before testing",
    )
    p_hub.add_argument("--exchange", default=None, help="Override exchange id")
    p_hub.add_argument(
        "--scan-file",
        type=Path,
        default=None,
        help="Optional existing scan_all parquet to summarize hub hit-rate",
    )

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
    alpha = config.cointegration.pvalue_threshold

    if args.command == "scan":
        df, sig = scan_pairs(
            config,
            symbols=args.symbols,
            exchange_id=args.exchange,
            universe_as_of=args.universe_as_of,
            persist=not args.no_persist,
            preprocess=args.preprocess,
        )
        if df.empty:
            print("No pairs tested (empty universe or insufficient overlap).")
            return 0
        cols = [
            "rank",
            "y_symbol",
            "x_symbol",
            "pvalue",
            "pvalue_fdr",
            "significant_raw",
            "significant_fdr",
            "hedge_ratio",
            "half_life",
            "n_obs",
        ]
        cols = [c for c in cols if c in df.columns]
        n_raw = int(df["significant_raw"].sum()) if "significant_raw" in df.columns else 0
        n_fdr = int(df["significant_fdr"].sum()) if "significant_fdr" in df.columns else 0
        print("=== ALL PAIRS (sorted by FDR p-value) ===")
        print(df[cols].to_string(index=False))
        print(
            f"\nTested: {len(df)}  |  raw p<{alpha}: {n_raw}  |  "
            f"BH-FDR p<{alpha}: {n_fdr}  |  naive E[FP]≈{0.05 * len(df):.1f}"
        )
        if n_fdr:
            print("\n=== FDR SURVIVORS ===")
            print(df.loc[df["significant_fdr"], cols].to_string(index=False))
        else:
            print("\n=== FDR SURVIVORS ===\n(none)")
        if n_raw and not n_fdr:
            print("\n=== RAW p<α (did NOT survive FDR) ===")
            print(df.loc[df["significant_raw"], cols].head(20).to_string(index=False))
        return 0

    if args.command == "pair":
        if args.sync:
            pipe = DataPipeline(config)
            pipe.sync_symbol(args.y, force_full=True)
            pipe.sync_symbol(args.x, force_full=True)
            eid = pipe.exchange_id
        else:
            eid = args.exchange or resolve_exchange_id(config)

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

    if args.command == "hub-test":
        if args.sync_placebos:
            pipe = DataPipeline(config)
            symbols = [args.hub, *(args.placebos or [])]
            if args.placebos is None:
                from pairs_trading.cointegration.hub import DEFAULT_PLACEBOS

                symbols = [args.hub, *DEFAULT_PLACEBOS]
            for sym in dict.fromkeys(symbols):
                try:
                    pipe.sync_symbol(sym, force_full=True)
                except Exception:
                    logging.exception("Failed syncing %s", sym)
            eid = pipe.exchange_id
        else:
            eid = args.exchange or resolve_exchange_id(config)

        placebo_df = run_hub_placebo_tests(
            config,
            hub_symbol=args.hub,
            placebos=args.placebos,
            exchange_id=eid,
        )
        if placebo_df.empty:
            print("No placebo pairs tested (missing data).")
            return 1

        save_scan_results(
            placebo_df,
            root=config.storage.root,
            exchange_id=eid,
            timeframe=config.ohlcv.timeframe,
            mode=config.cointegration.persist_mode,
            meta={"hub": args.hub, "kind": "hub_placebo"},
            basename="hub_placebo",
        )

        cols = [
            c
            for c in [
                "rank",
                "y_symbol",
                "x_symbol",
                "pvalue",
                "pvalue_fdr",
                "significant_raw",
                "significant_fdr",
                "hedge_ratio",
                "half_life",
                "n_obs",
            ]
            if c in placebo_df.columns
        ]
        n_raw = int(placebo_df["significant_raw"].sum())
        n_fdr = int(placebo_df["significant_fdr"].sum())
        print(f"=== HUB PLACEBO TESTS ({args.hub}) ===")
        print(placebo_df[cols].to_string(index=False))
        print(
            f"\nPlacebos tested: {len(placebo_df)}  |  raw p<{alpha}: {n_raw}  |  "
            f"BH-FDR p<{alpha}: {n_fdr}"
        )
        if n_raw:
            print(
                "\nInterpretation: hub shows raw-significant placebo hits → "
                "treat hub-containing universe hits with skepticism."
            )
        else:
            print(
                "\nInterpretation: no raw-significant placebo hits at α — "
                "hub contamination less supported by this placebo set."
            )

        scan_path = args.scan_file
        if scan_path is None:
            cand = (
                Path(config.storage.root)
                / eid
                / "cointegration"
                / f"scan_all_latest_{config.ohlcv.timeframe}.parquet"
            )
            if cand.is_file():
                scan_path = cand
        if scan_path and Path(scan_path).is_file():
            import pandas as pd

            scan_df = pd.read_parquet(scan_path)
            if "significant_raw" not in scan_df.columns and "pvalue" in scan_df.columns:
                from pairs_trading.cointegration.fdr import apply_benjamini_hochberg

                scan_df = apply_benjamini_hochberg(scan_df, alpha=alpha)
            summary = summarize_hub_from_scan(scan_df, hub_symbol=args.hub)
            print("\n=== HUB HIT-RATE ON UNIVERSE SCAN ===")
            print(json.dumps(summary, indent=2))
        return 0

    build_parser().error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
