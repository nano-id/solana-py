"""CLI for Module 3 factor-beta research (step-gated)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from pairs_trading.factor_beta.config import load_factor_beta_config
from pairs_trading.factor_beta.data import (
    alignment_transparency,
    inventory_report,
    save_inventory,
)
from pairs_trading.factor_beta.factor_model import run_step2_all_lookbacks


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
        # Drop bulky default_report from stdout; write already done inside
        slim = {k: v for k, v in bundle.items() if k != "default_report"}
        print(json.dumps(slim, indent=2))
        print(f"\nWrote under {out}", file=sys.stderr)
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
