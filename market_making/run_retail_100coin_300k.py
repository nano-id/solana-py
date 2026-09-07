"""Run retail $3500/slot MM on 100 coins with $300k wallet (85 slots)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from market_making.config import load_mm_config
from market_making.multi_portfolio import (
    run_multi_portfolio_range,
    save_multi_portfolio_artifacts,
)

PROFILE = {
    "label": "retail_100coin_300k_v1",
    "config": "config/retail_100coin_300k_v1.yaml",
    "equity": 300_000.0,
    "max_slots": 85,
    "slot_notional": 3_500.0,
}

WINDOWS = (("2026-01-01", "2026-09-06", "2026ytd"),)


def main() -> int:
    cfg = load_mm_config(PROFILE["config"])
    runs: list[dict[str, Any]] = []
    for start, end, tag in WINDOWS:
        label = PROFILE["label"]
        print(f"Running {label} {tag} ({start}→{end}) ...", flush=True)
        result, frame = run_multi_portfolio_range(
            cfg,
            start=start,
            end_exclusive=end,
            timeframe="1h",
            seed=42,
            initial_equity=float(PROFILE["equity"]),
            max_slots=int(PROFILE["max_slots"]),
            slot_notional_usd=float(PROFILE["slot_notional"]),
        )
        paths = save_multi_portfolio_artifacts(
            result,
            frame,
            root=cfg.storage_root,
            exchange_id=cfg.exchange_id,
            basename=f"{label}_{tag}",
        )
        slim = asdict(result)
        runs.append(
            {
                "label": label,
                "tag": tag,
                "window": f"{start}→{end}",
                "n_symbols": len(cfg.symbols),
                "result": slim,
                "artifacts": paths,
            }
        )
        active = sum(1 for v in slim["active_symbol_bars"].values() if v > 0)
        print(
            f"  → equity {slim['start_equity']:.0f}→{slim['final_equity']:.2f} "
            f"({slim['total_return_pct']:.2f}%) entries={slim['n_entries']} "
            f"skipped_busy={slim['n_skipped_busy']} "
            f"active_syms={active}/{len(cfg.symbols)} fees={slim['fees_paid']:.2f}",
            flush=True,
        )

    summary = {
        "interpretation_allowed": False,
        "design": {
            "universe": "top100_okx_volume_with_full_2026_1h_coverage",
            "wallet": PROFILE["equity"],
            "max_slots": PROFILE["max_slots"],
            "slot_notional": PROFILE["slot_notional"],
            "params": "retail hourly AS v2 (gamma=0.18, kappa=0.8, tau=48, fill=0.25)",
            "n_symbols": len(cfg.symbols),
            "compare_to": ["retail_100coin_35k_v1", "retail_100coin_100k_v1"],
        },
        "runs": [
            {
                "label": r["label"],
                "tag": r["tag"],
                "window": r["window"],
                "n_symbols": r["n_symbols"],
                "start_equity": r["result"]["start_equity"],
                "final_equity": r["result"]["final_equity"],
                "total_return_pct": r["result"]["total_return_pct"],
                "n_entries": r["result"]["n_entries"],
                "n_round_trips": r["result"]["n_round_trips"],
                "n_skipped_busy": r["result"]["n_skipped_busy"],
                "fees_paid": r["result"]["fees_paid"],
                "n_active_symbols": sum(
                    1 for v in r["result"]["active_symbol_bars"].values() if v > 0
                ),
                "active_symbol_bars": r["result"]["active_symbol_bars"],
            }
            for r in runs
        ],
    }
    art = Path("/opt/cursor/artifacts")
    art.mkdir(parents=True, exist_ok=True)
    (art / "retail_100coin_300k_2026ytd_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    Path("config/retail_100coin_300k_results.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print("Wrote config/retail_100coin_300k_results.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
