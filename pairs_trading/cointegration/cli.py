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
    p_joh.add_argument(
        "--mc-block-bootstrap",
        action="store_true",
        help="Also run correlation-preserving block-bootstrap null (needs local OHLCV)",
    )
    p_joh.add_argument(
        "--mc-block-size",
        type=int,
        default=None,
        help="Block length for return-vector bootstrap (default: sqrt(T))",
    )

    p_dual = sub.add_parser(
        "johansen-dual-null",
        help="Compare indep-RW vs block-bootstrap nulls for explicit baskets",
    )
    p_dual.add_argument(
        "--baskets",
        nargs="+",
        required=True,
        help="Baskets as pipe-joined symbols, e.g. 'BNB/USDT|BTC/USDT|ETH/USDT'",
    )
    p_dual.add_argument("--exchange", default=None)
    p_dual.add_argument("--mc-sims", type=int, default=2000)
    p_dual.add_argument("--seed", type=int, default=42)
    p_dual.add_argument("--mc-block-size", type=int, default=None)
    p_dual.add_argument("--no-persist", action="store_true")
    p_dual.add_argument(
        "--basename",
        default="johansen_dual_null",
        help="Artifact basename when persisting",
    )

    p_eg_dual = sub.add_parser(
        "eg-dual-null",
        help="Dual-null EG (indep-RW vs block-bootstrap) on rolling FDR survivors",
    )
    p_eg_dual.add_argument("--exchange", default=None)
    p_eg_dual.add_argument(
        "--window-days",
        nargs="+",
        type=int,
        default=[180, 365],
        help="Which rolling_*d_none artifacts to load",
    )
    p_eg_dual.add_argument("--mc-sims", type=int, default=2000)
    p_eg_dual.add_argument("--seed", type=int, default=42)
    p_eg_dual.add_argument("--mc-block-size", type=int, default=None)
    p_eg_dual.add_argument("--no-persist", action="store_true")

    p_resid = sub.add_parser(
        "residual-retest",
        help="BNB-free candidates: BTC + alt-index residual EG; BY on EG p_bb",
    )
    p_resid.add_argument("--exchange", default=None)
    p_resid.add_argument("--no-persist", action="store_true")

    p_wld = sub.add_parser(
        "wld-hub",
        help="WLD hub/placebo rolling EG + dual-null; residual corr of WLD candidates",
    )
    p_wld.add_argument("--exchange", default=None)
    p_wld.add_argument("--window-days", type=int, default=180)
    p_wld.add_argument("--step-days", type=int, default=90)
    p_wld.add_argument("--mc-sims", type=int, default=1500)
    p_wld.add_argument("--seed", type=int, default=42)
    p_wld.add_argument(
        "--placebos",
        nargs="+",
        default=None,
        help="Default: LTC ATOM NEAR DOGE XAUT",
    )
    p_wld.add_argument("--no-persist", action="store_true")

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
            store = ParquetStore(
                config.storage.root, eid, config.ohlcv.timeframe
            )
            enriched = enrich_top_baskets_monte_carlo(
                df,
                top_n=args.mc_top,
                n_sims=args.mc_sims,
                seed=args.seed,
                block_bootstrap=bool(args.mc_block_bootstrap),
                block_size=args.mc_block_size,
                store=store if args.mc_block_bootstrap else None,
                min_obs=config.cointegration.min_obs,
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
                    "pvalue_bb",
                    "p_ratio_bb_over_rw",
                    "mc_null_p95",
                    "bb_null_p95",
                    "block_size",
                    "mc_n_sims",
                    "contains_bnb",
                ]
                if c in enriched.columns
            ]
            mc_view = enriched[enriched["pvalue_mc"].notna()]
            label = (
                f"DUAL NULL (indep-RW vs block-bootstrap, top {args.mc_top}, "
                f"{args.mc_sims} sims)"
                if args.mc_block_bootstrap
                else f"MONTE CARLO p (top {args.mc_top} by trace, {args.mc_sims} sims)"
            )
            print(f"\n=== {label} ===")
            print(mc_view[mc_cols].to_string(index=False))
            if args.mc_block_bootstrap and "pvalue_bb" in mc_view.columns:
                ratios = mc_view["p_ratio_bb_over_rw"].dropna()
                if not ratios.empty:
                    print(
                        f"\np_bb / p_rw: median={ratios.median():.2f}  "
                        f"min={ratios.min():.2f}  max={ratios.max():.2f}  "
                        f"(>1 ⇒ correlation-preserving null less significant)"
                    )
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
                        "block_bootstrap": bool(args.mc_block_bootstrap),
                        "block_size": args.mc_block_size or "sqrt_T",
                    },
                    basename=(
                        "johansen_bnb_free_mc"
                        if args.bnb_free_random
                        else "johansen_baskets_mc"
                    ),
                )
        return 0

    if args.command == "johansen-dual-null":
        from pairs_trading.cointegration.johansen_mc import compare_dual_nulls_for_baskets

        baskets = [b.split("|") for b in args.baskets]
        df = compare_dual_nulls_for_baskets(
            baskets,
            config,
            exchange_id=args.exchange,
            n_sims=args.mc_sims,
            seed=args.seed,
            block_size=args.mc_block_size,
            persist=not args.no_persist,
            basename=args.basename,
        )
        cols = [
            c
            for c in [
                "symbols",
                "contains_bnb",
                "trace_r0",
                "cv95_r0",
                "pvalue_table",
                "pvalue_mc",
                "pvalue_bb",
                "p_ratio_bb_over_rw",
                "mc_null_p95",
                "bb_null_p95",
                "block_size",
                "n_sims",
            ]
            if c in df.columns
        ]
        print("=== JOHANSEN DUAL NULL (indep-RW vs block-bootstrap) ===")
        print(df[cols].to_string(index=False))
        ratios = df["p_ratio_bb_over_rw"].dropna()
        if not ratios.empty:
            print(
                f"\np_bb / p_rw: median={ratios.median():.2f}  "
                f"min={ratios.min():.2f}  max={ratios.max():.2f}"
            )
            print(
                "Interpretation: ratio ≫ 1 means common-factor / correlation "
                "structure made the independent-RW null too optimistic."
            )
        return 0

    if args.command == "eg-dual-null":
        from pairs_trading.cointegration.eg_dual_null import (
            dual_null_rolling_fdr_pairs,
            summarize_eg_dual_by_bnb,
        )

        df = dual_null_rolling_fdr_pairs(
            config,
            exchange_id=args.exchange,
            window_days=tuple(args.window_days),
            n_sims=args.mc_sims,
            seed=args.seed,
            block_size=args.mc_block_size,
            persist=not args.no_persist,
        )
        if df.empty:
            print("No rolling FDR hits / dual-null rows.")
            return 1
        cols = [
            c
            for c in [
                "window_start",
                "window_end",
                "window_days",
                "pair",
                "contains_bnb",
                "pvalue_table",
                "pvalue_mc",
                "pvalue_bb",
                "p_ratio_bb_over_rw",
                "test_statistic",
                "rw_null_p05",
                "bb_null_p05",
                "block_size",
                "n_obs",
            ]
            if c in df.columns
        ]
        print("=== EG DUAL NULL ON ROLLING FDR SURVIVORS ===")
        print(df[cols].to_string(index=False))
        summary = summarize_eg_dual_by_bnb(df)
        print("\n=== BY GROUP (BNB vs BNB-free) ===")
        print(summary.to_string(index=False))
        print(
            "\nInterpretation: p_bb/p_rw ≫ 1 ⇒ correlation/common-factor made "
            "indep-RW (≈ MacKinnon) null too optimistic."
        )
        return 0

    if args.command == "residual-retest":
        from pairs_trading.cointegration.residual_retest import (
            run_p_bb_by_and_residual_report,
        )

        report = run_p_bb_by_and_residual_report(
            config,
            exchange_id=args.exchange,
            persist=not args.no_persist,
        )
        by_summary = report["by_summary"]
        all_by = report["p_bb_by_all"]
        uniq = report["p_bb_by_unique"]
        resid = report["residual_retest"]

        print("=== BY ON EG block-bootstrap p_bb (rolling FDR) ===")
        print(
            f"All rows: n={by_summary['n_all']}  "
            f"p_bb<α: {by_summary['n_all_bb_lt_alpha']}  "
            f"BY sig: {by_summary['n_all_by_sig']}"
        )
        print(
            f"Unique pairs (min p_bb): n={by_summary['n_unique']}  "
            f"p_bb<α: {by_summary['n_unique_bb_lt_alpha']}  "
            f"BY sig: {by_summary['n_unique_by_sig']}"
        )
        cols_all = [
            c
            for c in [
                "window_start",
                "window_end",
                "pair",
                "contains_bnb",
                "pvalue_bb",
                "pvalue_bb_by",
                "significant_bb_by",
            ]
            if c in all_by.columns
        ]
        print("\n--- all 13 ---")
        print(all_by[cols_all].to_string(index=False))
        cols_u = [
            c
            for c in [
                "pair_canon",
                "window_start",
                "window_end",
                "pair",
                "contains_bnb",
                "pvalue_bb",
                "pvalue_bb_by_unique",
                "significant_bb_by_unique",
            ]
            if c in uniq.columns
        ]
        print("\n--- unique pairs ---")
        print(uniq[cols_u].to_string(index=False))

        print("\n=== BNB-FREE CANDIDATES: raw / BTC / BTC+alt residual ===")
        if resid.empty:
            print("(no residual rows)")
        else:
            cols_r = [
                c
                for c in [
                    "window_start",
                    "window_end",
                    "pair",
                    "preprocess",
                    "pvalue",
                    "test_statistic",
                    "half_life",
                    "n_obs",
                    "alt_index_n_members",
                    "y_beta_btc",
                    "y_beta_alt",
                    "x_beta_btc",
                    "x_beta_alt",
                ]
                if c in resid.columns
            ]
            print(resid[cols_r].to_string(index=False))
            # Pivot-like summary: p by preprocess
            wide = resid.pivot_table(
                index=["window_start", "window_end", "pair"],
                columns="preprocess",
                values="pvalue",
                aggfunc="first",
            )
            print("\n--- p-value wide ---")
            print(wide.to_string())
        return 0

    if args.command == "wld-hub":
        import json

        from pairs_trading.cointegration.wld_hub import (
            correlate_wld_candidate_residuals,
            rolling_hub_placebo_screen,
        )

        eg_df, dual_df, summary = rolling_hub_placebo_screen(
            config,
            hub_symbol="WLD/USDT",
            placebos=args.placebos,
            exchange_id=args.exchange,
            window_days=args.window_days,
            step_days=args.step_days,
            n_sims=args.mc_sims,
            seed=args.seed,
            persist=not args.no_persist,
        )
        print("=== WLD HUB / PLACEBO ROLLING EG ===")
        print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, float) or v == v}, indent=2, default=str))
        print(
            f"\nHit-rate WLD-pairs: {summary.get('hit_rate_hub', float('nan')):.3f}  "
            f"({summary.get('n_hub_raw_hits')}/{summary.get('n_hub_pair_windows')})  |  "
            f"WLD-free pairs: {summary.get('hit_rate_other', float('nan')):.3f}  "
            f"({summary.get('n_other_raw_hits')}/{summary.get('n_other_pair_windows')})  |  "
            f"ratio: {summary.get('hit_rate_ratio', float('nan')):.2f}"
        )
        if not eg_df.empty:
            hub_hits = eg_df[eg_df["contains_hub"] & eg_df["significant_raw"]]
            cols = [
                c
                for c in [
                    "window_start",
                    "window_end",
                    "pair",
                    "pvalue",
                    "pvalue_fdr",
                    "significant_fdr",
                    "half_life",
                ]
                if c in hub_hits.columns
            ]
            print("\n--- WLD raw hits ---")
            print(hub_hits[cols].to_string(index=False) if not hub_hits.empty else "(none)")
        if not dual_df.empty:
            dcols = [
                c
                for c in [
                    "window_start",
                    "window_end",
                    "pair",
                    "pvalue_table",
                    "pvalue_mc",
                    "pvalue_bb",
                    "p_ratio_bb_over_rw",
                    "significant_raw",
                ]
                if c in dual_df.columns
            ]
            print("\n=== DUAL-NULL ON WLD|placebo WINDOWS ===")
            print(dual_df[dcols].to_string(index=False))
            print(
                f"\nWLD dual: median p_rw={summary.get('dual_hub_median_p_rw')}  "
                f"median p_bb={summary.get('dual_hub_median_p_bb')}  "
                f"frac p_bb<0.05={summary.get('dual_hub_frac_bb_lt_05')}"
            )

        corr_report = correlate_wld_candidate_residuals(
            config, exchange_id=args.exchange, persist=not args.no_persist
        )
        corr = corr_report["corr"]
        print("\n=== OKB-WLD / ADA-WLD / ARB-WLD RESIDUAL CORR ===")
        off = corr[corr["pair_a"] < corr["pair_b"]]  # unique unordered pairs
        print(off.to_string(index=False))
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
