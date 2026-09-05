"""CLI for Module 3 factor-beta research (step-gated)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from pairs_trading.factor_beta.config import load_factor_beta_config
from pairs_trading.factor_beta.data import inventory_report, save_inventory


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
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = load_factor_beta_config(args.config)
    if args.command == "step1-data":
        report = inventory_report(cfg)
        path = save_inventory(report, args.out or Path(cfg.storage.output_dir))
        print(json.dumps(report, indent=2))
        print(f"\nWrote {path}", file=sys.stderr)
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
