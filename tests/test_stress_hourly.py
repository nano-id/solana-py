"""Smoke tests for hourly stress battery."""

from __future__ import annotations

from market_making.config import MarketMakingConfig
from market_making.stress_hourly import (
    StressScenario,
    build_stress_scenarios,
    run_stress_battery,
)


def test_build_scenarios_nonempty() -> None:
    cfg = MarketMakingConfig()
    sc = build_stress_scenarios(cfg)
    assert len(sc) >= 10
    assert sc[0].name == "baseline"


def test_stress_battery_tiny() -> None:
    cfg = MarketMakingConfig(
        symbols=["BTC/USDT", "ETH/USDT"],
        commission_pct=0.07,
        storage_root="data",
        exchange_id="okx",
    )
    scenarios = [
        StressScenario(
            name="baseline",
            risk="control",
            description="baseline",
            mutate=lambda c: c,
            months=[(2026, 4)],
        ),
        StressScenario(
            name="fill_hard",
            risk="fill_model",
            description="hard fills",
            mutate=lambda c: c.model_copy(
                update={"sim": c.sim.model_copy(update={"fill_prob": 0.10})}
            ),
            months=[(2026, 4)],
        ),
    ]
    result = run_stress_battery(
        cfg,
        months=[(2026, 4)],
        timeframe="1h",
        seed=1,
        initial_equity=1000.0,
        scenarios=scenarios,
    )
    assert result.interpretation_allowed is False
    assert len(result.cases) == 2
    assert result.cases[0].name == "baseline"
