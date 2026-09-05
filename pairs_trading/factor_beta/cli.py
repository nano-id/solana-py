"""CLI for Module 3 factor-beta research (step-gated)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from pairs_trading.factor_beta.audits import (
    audit_c1_selection_leakage,
    audit_c3_plateau_width,
    save_audit_report,
)
from pairs_trading.factor_beta.beta_regime import (
    analyze_beta_sign_flip,
    save_beta_regime_report,
)
from pairs_trading.factor_beta.c1_diagnostic import run_c1_diagnostic, save_c1_diagnostic
from pairs_trading.factor_beta.clustering import cluster_residuals, save_cluster_report
from pairs_trading.factor_beta.config import load_factor_beta_config
from pairs_trading.factor_beta.data import (
    alignment_transparency,
    inventory_report,
    save_inventory,
)
from pairs_trading.factor_beta.factor_model import run_step2_all_lookbacks
from pairs_trading.factor_beta.rule5d import run_rule5d, save_rule5d_report
from pairs_trading.factor_beta.signals import (
    run_cluster_signal_skeleton,
    run_synthetic_ou_mechanical_test,
    save_step4_report,
)
from pairs_trading.factor_beta.stationarity import run_adf_filter, save_adf_report
from pairs_trading.factor_beta.walkforward import (
    run_walk_forward_all_lines,
    save_walk_forward_report,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pairs-factor",
        description="Module 3: BTC-beta residual strategy (step-gated; no early claims).",
    )
    p.add_argument("--config", type=Path, default=Path("config/factor_beta.yaml"))
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    s1 = sub.add_parser("step1-data", help="Inventory log-price panel + lock Rule-5 config")
    s1.add_argument("--out", type=Path, default=None)

    s1b = sub.add_parser(
        "step1-coverage",
        help="Per-coin RAW start/end vs aligned panel (no forward-fill check)",
    )
    s1b.add_argument("--out", type=Path, default=None)

    s2 = sub.add_parser(
        "step2-factor",
        help="Rolling BTC-beta factor model (β lookback ≠ trade/z windows)",
    )
    s2.add_argument("--out", type=Path, default=None)

    s2b = sub.add_parser(
        "step2b-beta-regime",
        help="Diagnostic: WLD/NEAR β sign-flip vs R² and BTC vol (not a trade step)",
    )
    s2b.add_argument("--out", type=Path, default=None)
    s2b.add_argument(
        "--symbols",
        nargs="+",
        default=["WLD/USDT", "NEAR/USDT"],
        help="Symbols to inspect for β sign flips",
    )

    s3 = sub.add_parser(
        "step3-adf",
        help="ADF stationarity filter on residuals (candidate gate)",
    )
    s3.add_argument("--out", type=Path, default=None)

    s5e = sub.add_parser(
        "step5e-cluster",
        help="Rule 5e: cluster ADF survivors by residual |corr| (before Step 4)",
    )
    s5e.add_argument("--out", type=Path, default=None)

    s4 = sub.add_parser(
        "step4-signals",
        help="Z-score skeleton: one line per Rule-5e cluster (thresholds uncalibrated)",
    )
    s4.add_argument("--out", type=Path, default=None)
    s4.add_argument(
        "--skip-ou-test",
        action="store_true",
        help="Skip synthetic OU mechanical wiring check",
    )

    sc1 = sub.add_parser(
        "step-c1-diagnostic",
        help="C1 commonality + equal-weight composite residual vs ADA rep (pre-Step 6)",
    )
    sc1.add_argument("--out", type=Path, default=None)

    s6 = sub.add_parser(
        "step6-walkforward",
        help="Per-line walk-forward z calibration (Rule 5b hard gate + plateau)",
    )
    s6.add_argument("--out", type=Path, default=None)

    saud = sub.add_parser(
        "step-audits",
        help="C1 selection-leakage audit + C3 plateau-width audit (controls)",
    )
    saud.add_argument("--out", type=Path, default=None)

    s5d = sub.add_parser(
        "step5d-dual-null",
        help="Rule 5d: dual-null + Benjamini-Yekutieli on C1/C3 (skip C2)",
    )
    s5d.add_argument("--out", type=Path, default=None)
    s5d.add_argument("--n-sims", type=int, default=80)
    s5d.add_argument("--workers", type=int, default=4)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = load_factor_beta_config(args.config)
    out = Path(args.out) if getattr(args, "out", None) else Path(cfg.storage.output_dir)

    if args.command == "step1-data":
        report = inventory_report(cfg)
        path = save_inventory(report, out)
        print(json.dumps(report, indent=2))
        print(f"\nWrote {path}", file=sys.stderr)
        return 0

    if args.command == "step1-coverage":
        report = alignment_transparency(cfg)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "step1_raw_coverage.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        print(f"\nWrote {path}", file=sys.stderr)
        return 0

    if args.command == "step2-factor":
        bundle = run_step2_all_lookbacks(cfg)
        slim = {k: v for k, v in bundle.items() if k != "default_report"}
        print(json.dumps(slim, indent=2))
        print(f"\nWrote under {out}", file=sys.stderr)
        return 0

    if args.command == "step2b-beta-regime":
        report = analyze_beta_sign_flip(cfg, symbols=list(args.symbols))
        path = save_beta_regime_report(report, out)
        print(json.dumps(report, indent=2))
        print(f"\nWrote {path}", file=sys.stderr)
        return 0

    if args.command == "step3-adf":
        report = run_adf_filter(cfg)
        path = save_adf_report(report, out)
        print(json.dumps(report, indent=2))
        print(f"\nWrote {path}", file=sys.stderr)
        return 0

    if args.command == "step5e-cluster":
        report = cluster_residuals(cfg)
        path = save_cluster_report(report, out)
        print(json.dumps(report, indent=2))
        print(f"\nWrote {path}", file=sys.stderr)
        return 0

    if args.command == "step4-signals":
        report = run_cluster_signal_skeleton(cfg)
        if not args.skip_ou_test:
            report["synthetic_ou_mechanical"] = run_synthetic_ou_mechanical_test(cfg)
        path = save_step4_report(report, out)
        print(json.dumps(report, indent=2))
        print(f"\nWrote {path}", file=sys.stderr)
        return 0

    if args.command == "step-c1-diagnostic":
        report = run_c1_diagnostic(cfg)
        path = save_c1_diagnostic(report, out)
        print(json.dumps(report, indent=2))
        print(f"\nWrote {path}", file=sys.stderr)
        return 0

    if args.command == "step6-walkforward":
        report = run_walk_forward_all_lines(cfg)
        path = save_walk_forward_report(report, out)
        print(json.dumps(report, indent=2))
        print(f"\nWrote {path}", file=sys.stderr)
        return 0

    if args.command == "step-audits":
        c1 = audit_c1_selection_leakage(cfg)
        c3 = audit_c3_plateau_width(cfg)
        p1 = save_audit_report(c1, out, "step_c1_selection_leakage_audit.json")
        p3 = save_audit_report(c3, out, "step_c3_plateau_width_audit.json")
        bundle = {
            "interpretation_allowed": False,
            "c1_selection_leakage_audit": c1,
            "c3_plateau_width_audit": c3,
        }
        print(json.dumps(bundle, indent=2))
        print(f"\nWrote {p1}\nWrote {p3}", file=sys.stderr)
        return 0

    if args.command == "step5d-dual-null":
        report = run_rule5d(cfg, n_sims=int(args.n_sims), max_workers=int(args.workers))
        path = save_rule5d_report(report, out)
        print(json.dumps(report, indent=2))
        print(f"\nWrote {path}", file=sys.stderr)
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
