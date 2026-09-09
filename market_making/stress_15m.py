"""Robustness stress battery for the calibrated 15m MM parameter set.

Mirrors ``stress_hourly`` on the 15m timeframe. Blind window default:
2026-04 → 2026-08 (post Mar-2026 OOS). Not a live-edge claim.

Does not modify the frozen 1h package.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from market_making.config import MarketMakingConfig
from market_making.stress_hourly import (
    StressBatteryResult,
    build_stress_scenarios,
    run_stress_battery,
)


def default_15m_blind_months() -> list[tuple[int, int]]:
    return [(2026, 4), (2026, 5), (2026, 6), (2026, 7), (2026, 8)]


def run_15m_stress_battery(
    cfg: MarketMakingConfig,
    *,
    months: list[tuple[int, int]] | None = None,
    timeframe: str = "15m",
    seed: int = 42,
    initial_equity: float = 1000.0,
) -> StressBatteryResult:
    months = months or default_15m_blind_months()
    sim = cfg.sim.model_copy(update={"bars_per_year": 35040.0})
    cfg15 = cfg.model_copy(update={"sim": sim})
    result = run_stress_battery(
        cfg15,
        months=months,
        timeframe=timeframe,
        seed=seed,
        initial_equity=initial_equity,
        scenarios=build_stress_scenarios(cfg15),
    )
    result.method = (
        f"15m stress battery; window={months}; "
        "mirrors hourly risks on 15m candles. Not a live-edge claim."
    )
    result.timeframe = timeframe
    return result


def save_15m_stress(
    result: StressBatteryResult,
    *,
    root: str | Path = "data",
    exchange_id: str = "okx",
    basename: str = "stress_15m_blind_2026",
) -> dict[str, str]:
    out = Path(root) / exchange_id / "market_making"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{basename}.json"
    path.write_text(json.dumps(asdict(result), indent=2, default=str), encoding="utf-8")
    art = Path("/opt/cursor/artifacts")
    art.mkdir(parents=True, exist_ok=True)
    art_path = art / f"{basename}.json"
    art_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return {"summary": str(path), "artifact": str(art_path)}
