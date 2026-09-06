"""Run large-book profiles on the top-30 majors with no busy-slot skip.

``max_slots`` equals universe size; slot notional = wallet / n_symbols.
Paper only; interpretation_allowed=false.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from market_making.config import load_mm_config
from market_making.multi_portfolio import (
    run_multi_portfolio_range,
    save_multi_portfolio_artifacts,
)

PROFILES = (
    {
        "label": "large_book_25k_30coin_v1",
        "config": "config/large_book_25k_30coin_v1.yaml",
        "equity": 25_000.0,
        "max_slots": 30,
        "slot_notional": 25_000.0 / 30.0,
    },
    {
        "label": "large_book_50k_30coin_v1",
        "config": "config/large_book_50k_30coin_v1.yaml",
        "equity": 50_000.0,
        "max_slots": 30,
        "slot_notional": 50_000.0 / 30.0,
    },
)

WINDOWS = (
    ("2025-01-01", "2026-01-01", "2025full"),
    ("2026-01-01", "2026-09-06", "2026ytd"),
)


def _slim(result: Any) -> dict[str, Any]:
    d = asdict(result)
    # Keep active bars but drop huge frame elsewhere
    return d


def run_profile(
    profile: dict[str, Any],
    *,
    start: str,
    end_exclusive: str,
    tag: str,
    seed: int = 42,
    persist: bool = True,
) -> dict[str, Any]:
    cfg = load_mm_config(profile["config"])
    result, frame = run_multi_portfolio_range(
        cfg,
        start=start,
        end_exclusive=end_exclusive,
        timeframe="1h",
        seed=seed,
        initial_equity=float(profile["equity"]),
        max_slots=int(profile["max_slots"]),
        slot_notional_usd=float(profile["slot_notional"]),
    )
    paths: dict[str, str] = {}
    if persist:
        paths = save_multi_portfolio_artifacts(
            result,
            frame,
            root=cfg.storage_root,
            exchange_id=cfg.exchange_id,
            basename=f"{profile['label']}_{tag}",
        )
        art = Path("/opt/cursor/artifacts")
        art.mkdir(parents=True, exist_ok=True)
        # Mirror summary + equity curve for walkthrough
        summary_path = art / f"{profile['label']}_{tag}.json"
        csv_path = art / f"{profile['label']}_{tag}.csv"
        summary_path.write_text(json.dumps(_slim(result), indent=2, default=str))
        frame.to_csv(csv_path, index=False)
        paths["artifact_json"] = str(summary_path)
        paths["artifact_csv"] = str(csv_path)
    return {
        "label": profile["label"],
        "window": f"{start}→{end_exclusive}",
        "tag": tag,
        "n_symbols": len(cfg.symbols),
        "symbols": list(cfg.symbols),
        "equity0": profile["equity"],
        "max_slots": profile["max_slots"],
        "slot_notional": profile["slot_notional"],
        "result": _slim(result),
        "artifacts": paths,
    }


def main() -> int:
    out: dict[str, Any] = {
        "interpretation_allowed": False,
        "design": {
            "universe": "top30_majors",
            "max_slots": 30,
            "busy_slot_skip": False,
            "slot_notional_rule": "wallet / 30",
            "params": "large_book_v1 AS (gamma=0.25, kappa=0.6, tau=72, ...)",
        },
        "runs": [],
    }
    for profile in PROFILES:
        for start, end, tag in WINDOWS:
            print(
                f"Running {profile['label']} {tag} "
                f"({start}→{end}) ...",
                flush=True,
            )
            out["runs"].append(run_profile(profile, start=start, end_exclusive=end, tag=tag))
            r = out["runs"][-1]["result"]
            print(
                f"  → equity {r['start_equity']:.0f}→{r['final_equity']:.2f} "
                f"({r['total_return_pct']:.2f}%) entries={r['n_entries']} "
                f"skipped_busy={r['n_skipped_busy']} "
                f"active_syms={sum(1 for v in r['active_symbol_bars'].values() if v > 0)}/30",
                flush=True,
            )

    summary_path = Path("/opt/cursor/artifacts/large_book_30coin_runs.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(out, indent=2, default=str))
    repo_path = Path("data/okx/market_making/large_book_30coin_runs.json")
    repo_path.parent.mkdir(parents=True, exist_ok=True)
    repo_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote {summary_path} and {repo_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
