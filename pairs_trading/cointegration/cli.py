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

    p_roll = sub.add_parser("rolling", help="Rolling-window EG+FDR screen")
    p_roll.add_argument("--window-days", type=int, default=180)
    p_roll.add_argument("--step-days", type=int, default=90)
    p_roll.add_argument("--exchange", default=None)
    p_roll.add_argument("--symbols", nargs="+", default=None)
    p_roll.add_argument(
        "--preprocess",
        choices=["none", "btc_residual"],
        default="none",
    )
    p_roll.add_argument("--no-persist", action="store_true")

    p_joh = sub.add_parser("johansen", help="Johansen multivariate basket screen + BH-FDR")
    p_joh.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Candidate symbols (default: BNB OKB SOL UNI ARB WLD)",
    )
    p_joh.add_argument("--sizes", nargs="+", type=int, default=[3, 4])
    p_joh.add_argument("--exchange", default=None)
    p_joh.add_argument("--no-persist", action="store_true")
    p_joh.add_argument(
        "--bnb-free-random",
        action="store_true",
        help="Baseline: random 3/4-tuples from BNB-free local universe",
    )
    p_joh.add_argument(
        "--n-random",
        type=int,
        default=20,
        help="Number of random BNB-free baskets (with --bnb-free-random)",
    )
    p_joh.add_argument("--seed", type=int, default=42)
    p_joh.add_argument(
        "--mc-top",
        type=int,
        default=0,
        help="After scan, Monte Carlo p-values for top-N baskets by trace (0=skip)",
    )
    p_joh.add_argument("--mc-sims", type=int, default=2000, help="MC simulations per basket")

    p_hf = sub.add_parser(
        "hf-retest",
        help="Retest daily rolling FDR survivors at higher frequency (default 4h)",
    )
    p_hf.add_argument("--hf-timeframe", default="4h")
    p_hf.add_argument("--exchange", default=None)
    p_hf.add_argument(
        "--window-days",
        nargs="+",
        type=int,
        default=[180, 365],
        help="Which rolling_*d_none artifacts to load",
    )
    p_hf.add_argument("--no-persist", action="store_true")

    p_sig = sub.add_parser("signal-demo", help="Phase 3 skeleton on synthetic OU spread")
    p_sig.add_argument("--n", type=int, default=2000)
    p_sig.add_argument("--seed", type=int, default=42)
    p_sig.add_argument("--entry-z", type=float, default=2.0)
    p_sig.add_argument("--exit-z", type=float, default=0.5)
    p_sig.add_argument("--stop-z", type=float, default=4.0)

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

    if args.command == "rolling":
        from pairs_trading.cointegration.rolling import scan_rolling_windows, summarize_rolling

        df = scan_rolling_windows(
            config,
            window_days=args.window_days,
            step_days=args.step_days,
            exchange_id=args.exchange,
            symbols=args.symbols,
            preprocess=args.preprocess,
            persist=not args.no_persist,
        )
        if df.empty:
            print("No rolling results.")
            return 1
        summary = summarize_rolling(df)
        print("=== ROLLING WINDOW SUMMARY ===")
        print(summary.to_string(index=False))
        fdr_hits = df[df["significant_fdr"]]
        print(
            f"\nWindows: {summary.shape[0]}  |  "
            f"total FDR hits (row-level): {len(fdr_hits)}  |  "
            f"windows with ≥1 FDR hit: {int((summary['n_fdr'] > 0).sum())}"
        )
        if not fdr_hits.empty:
            cols = [
                c
                for c in [
                    "window_start",
                    "window_end",
                    "y_symbol",
                    "x_symbol",
                    "pvalue",
                    "pvalue_fdr",
                    "hedge_ratio",
                    "half_life",
                ]
                if c in fdr_hits.columns
            ]
            print("\n=== FDR SURVIVORS BY WINDOW ===")
            print(fdr_hits[cols].to_string(index=False))
        else:
            print("\n=== FDR SURVIVORS BY WINDOW ===\n(none)")
        return 0

    if args.command == "johansen":
        from pairs_trading.cointegration.johansen import (
            scan_johansen_baskets,
            scan_johansen_random_bnb_free,
        )

        if args.bnb_free_random:
            df = scan_johansen_random_bnb_free(
                config,
                n_baskets=args.n_random,
                sizes=tuple(args.sizes),
                seed=args.seed,
                exchange_id=args.exchange,
                persist=not args.no_persist,
            )
            title = "JOHANSEN BNB-FREE RANDOM BASELINE"
        else:
            df = scan_johansen_baskets(
                config,
                candidates=args.symbols,
                sizes=tuple(args.sizes),
                exchange_id=args.exchange,
                persist=not args.no_persist,
            )
            title = "JOHANSEN BASKETS (+ BH-FDR)"
        if df.empty:
            print("No Johansen results.")
            return 1
        cols = [
            c
            for c in [
                "symbols",
                "n_symbols",
                "n_obs",
                "n_coint_relations_95",
                "has_coint_95",
                "pvalue",
                "pvalue_fdr",
                "significant_fdr",
                "pvalue_by",
                "significant_by",
                "contains_bnb",
                "trace_r0",
                "cv95_r0",
            ]
            if c in df.columns
        ]
        print(f"=== {title} ===")
        print(df[cols].to_string(index=False))
        n_hit = int(df["has_coint_95"].sum())
        n_fdr = int(df["significant_fdr"].sum()) if "significant_fdr" in df.columns else 0
        n_by = int(df["significant_by"].sum()) if "significant_by" in df.columns else 0
        print(
            f"\nBaskets tested: {len(df)}  |  raw ≥1 coint @95%: {n_hit} "
            f"({n_hit / len(df):.0%})  |  BH-FDR: {n_fdr} ({n_fdr / len(df):.0%})  |  "
            f"BY-FDR: {n_by} ({n_by / len(df):.0%})"
        )
        print(
            "\n(BY = Benjamini–Yekutieli: valid under arbitrary dependence; "
            "stricter than BH when tests overlap.)"
        )
        if n_fdr and "significant_fdr" in df.columns:
            print("\n=== BH-FDR SURVIVORS ===")
            print(df.loc[df["significant_fdr"], cols].to_string(index=False))
        if n_by and "significant_by" in df.columns:
            print("\n=== BY-FDR SURVIVORS ===")
            print(df.loc[df["significant_by"], cols].to_string(index=False))
            if "contains_bnb" in df.columns:
                no_bnb = df.loc[df["significant_by"] & ~df["contains_bnb"]]
                print(f"\nBY survivors without BNB: {len(no_bnb)}")
        elif "significant_by" in df.columns:
            print("\n=== BY-FDR SURVIVORS ===\n(none)")

        if args.mc_top and args.mc_top > 0:
            from pairs_trading.cointegration.johansen_mc import enrich_top_baskets_monte_carlo
            from pairs_trading.cointegration.persist import save_scan_results as _save

            eid = args.exchange or resolve_exchange_id(config)
            enriched = enrich_top_baskets_monte_carlo(
                df, top_n=args.mc_top, n_sims=args.mc_sims, seed=args.seed
            )
            mc_cols = [
                c
                for c in [
                    "symbols",
                    "n_symbols",
                    "n_obs",
                    "trace_r0",
                    "cv95_r0",
                    "pvalue_table",
                    "pvalue_mc",
                    "mc_null_p95",
                    "mc_null_p99",
                    "mc_n_sims",
                    "contains_bnb",
                ]
                if c in enriched.columns
            ]
            mc_view = enriched[enriched["pvalue_mc"].notna()]
            print(f"\n=== MONTE CARLO p (top {args.mc_top} by trace, {args.mc_sims} sims) ===")
            print(mc_view[mc_cols].to_string(index=False))
            if not args.no_persist and not mc_view.empty:
                kind = "bnb_free" if args.bnb_free_random else "baskets"
                _save(
                    enriched,
                    root=config.storage.root,
                    exchange_id=eid,
                    timeframe=config.ohlcv.timeframe,
                    mode=config.cointegration.persist_mode,
                    meta={
                        "kind": "johansen_monte_carlo",
                        "scan": kind,
                        "top_n": args.mc_top,
                        "n_sims": args.mc_sims,
                        "seed": args.seed,
                    },
                    basename=(
                        "johansen_bnb_free_mc"
                        if args.bnb_free_random
                        else "johansen_baskets_mc"
                    ),
                )
        return 0

    if args.command == "hf-retest":
        from pairs_trading.cointegration.hf_retest import (
            load_rolling_fdr_hits,
            retest_daily_fdr_pairs_at_hf,
        )

        hits = load_rolling_fdr_hits(
            config,
            exchange_id=args.exchange,
            window_days=args.window_days,
            timeframe="1d",
        )
        if hits.empty:
            print("No daily rolling FDR hits found on disk.")
            return 1
        df = retest_daily_fdr_pairs_at_hf(
            hits,
            config,
            hf_timeframe=args.hf_timeframe,
            exchange_id=args.exchange,
            persist=not args.no_persist,
        )
        if df.empty:
            print(
                f"HF retest produced no rows "
                f"(missing {args.hf_timeframe} parquet? sync first)."
            )
            return 1
        cols = [
            c
            for c in [
                "window_start",
                "window_end",
                "window_days",
                "pair",
                "pvalue",
                "daily_pvalue",
                "daily_half_life_days",
                "hf_half_life_bars",
                "hf_half_life_days",
                "hl_ratio_hf_over_daily",
                "n_obs",
            ]
            if c in df.columns
        ]
        print(f"=== HF RETEST ({args.hf_timeframe}) OF DAILY FDR HITS ===")
        print(df[cols].to_string(index=False))
        n_raw = int((df["pvalue"] < alpha).sum())
        print(
            f"\nRetested: {len(df)}  |  still raw p<{alpha} at {args.hf_timeframe}: "
            f"{n_raw}/{len(df)}"
        )
        return 0

    if args.command == "signal-demo":
        from pairs_trading.signal import SignalConfig, generate_signals, simulate_ou_spread, summarize_signals

        spread = simulate_ou_spread(n=args.n, seed=args.seed)
        cfg = SignalConfig(entry_z=args.entry_z, exit_z=args.exit_z, stop_z=args.stop_z)
        frame, events = generate_signals(spread, cfg)
        summary = summarize_signals(frame, events)
        print("=== SYNTHETIC OU SIGNAL DEMO ===")
        print(json.dumps(summary, indent=2))
        print("\nFirst 10 events:")
        for e in events[:10]:
            print(f"  bar={e.bar} z={e.z:.3f} reason={e.reason} side={e.side.value}")
        return 0

    build_parser().error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
