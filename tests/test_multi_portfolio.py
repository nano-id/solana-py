"""Smoke test for multi-slot portfolio."""

from __future__ import annotations

from market_making.config import MarketMakingConfig
from market_making.multi_portfolio import run_multi_portfolio_range


def test_multi_portfolio_smoke() -> None:
    cfg = MarketMakingConfig(
        symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"],
        commission_pct=0.07,
        storage_root="data",
        exchange_id="okx",
    )
    cfg = cfg.model_copy(
        update={"sim": cfg.sim.model_copy(update={"fill_prob": 0.25, "adverse_frac": 0.55})}
    )
    result, frame = run_multi_portfolio_range(
        cfg,
        start="2026-04-01",
        end_exclusive="2026-05-01",
        timeframe="1h",
        seed=1,
        initial_equity=10_500.0,
        max_slots=3,
        slot_notional_usd=3_500.0,
    )
    assert result.interpretation_allowed is False
    assert result.max_slots == 3
    assert result.slot_notional_usd == 3500.0
    assert result.n_bars > 0
    assert not frame.empty
    assert frame["n_open_slots"].max() <= 3
