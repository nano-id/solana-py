"""Unit tests for spot market making (no network)."""

from __future__ import annotations

import numpy as np
import pytest

from market_making.config import MarketMakingConfig, load_mm_config
from market_making.engine import QuoteEngine
from market_making.fees import (
    fee_on_notional,
    is_positive_half_spread,
    round_trip_net_edge_bps,
)
from market_making.inventory import InventoryState
from market_making.simulator import run_paper_sim


def test_commission_0_07_is_seven_bps() -> None:
    cfg = MarketMakingConfig(commission_pct=0.07)
    assert abs(cfg.commission_bps - 7.0) < 1e-9
    assert abs(cfg.commission_frac - 0.0007) < 1e-12


def test_required_half_spread_clears_fee_plus_buffer() -> None:
    cfg = MarketMakingConfig(
        commission_pct=0.07, edge_buffer_bps=5.0, min_half_spread_bps=12.0
    )
    # max(12, 7+5) = 12
    assert cfg.required_half_spread_bps == 12.0
    assert is_positive_half_spread(12.0, cfg)
    assert not is_positive_half_spread(11.0, cfg)
    assert round_trip_net_edge_bps(12.0, cfg) == pytest.approx(10.0)


def test_fee_on_notional() -> None:
    cfg = MarketMakingConfig(commission_pct=0.07)
    assert fee_on_notional(10_000.0, cfg) == pytest.approx(7.0)


def test_quote_engine_refuses_when_halted() -> None:
    cfg = MarketMakingConfig(commission_pct=0.07, max_inventory=1.0)
    eng = QuoteEngine(cfg)
    inv = InventoryState(
        base=0.96, quote=100.0, target=0.0, max_abs=1.0, halt_frac=0.95
    )
    q = eng.quote(100.0, inv, mid_history=np.array([100.0]))
    assert not q.quoting
    assert q.reason == "inventory_halt"


def test_quote_at_flat_is_positive_edge() -> None:
    cfg = MarketMakingConfig(commission_pct=0.07)
    eng = QuoteEngine(cfg)
    inv = InventoryState(base=0.0, quote=200.0, target=0.0, max_abs=1.0)
    q = eng.quote(100.0, inv, mid_history=np.array([100.0]))
    assert q.quoting
    assert q.half_spread_bps >= cfg.required_half_spread_bps
    assert q.net_edge_bps_if_matched > 0
    assert q.bid < q.mid < q.ask


def test_inventory_buy_sell_round_trip_realizes() -> None:
    inv = InventoryState(base=0.0, quote=1000.0)
    inv.apply_buy(price=100.0, size=1.0, fee=0.07, bar=0)
    inv.apply_sell(price=101.0, size=1.0, fee=0.07, bar=1)
    assert abs(inv.base) < 1e-12
    assert inv.realized_pnl == pytest.approx(1.0)
    assert inv.n_buys == 1 and inv.n_sells == 1


def test_load_yaml_and_paper_sim_net_positive() -> None:
    cfg = load_mm_config("config/market_making.yaml")
    assert cfg.commission_pct == 0.07
    assert len(cfg.symbols) >= 10
    assert all(":" not in s for s in cfg.symbols)  # spot only, no perp colon
    cfg.sim.n_bars = 1500
    cfg.sim.seed = 42
    result, frame = run_paper_sim(cfg)
    assert result.n_buys + result.n_sells > 0
    assert result.avg_half_spread_bps >= cfg.required_half_spread_bps
    assert result.interpretation_allowed is False
    # Under this paper fill model + positivity gate, equity should rise.
    assert result.net_positive is True
    assert frame["equity"].iloc[-1] > frame["equity"].iloc[0]
    assert "buy_size" in frame.columns and "sell_size" in frame.columns
    assert float(frame.loc[frame["buy_fill"], "buy_size"].min()) > 0
    assert float(frame.loc[frame["sell_fill"], "sell_size"].min()) > 0


def test_reject_fraction_commission_typo() -> None:
    with pytest.raises(ValueError):
        MarketMakingConfig(commission_pct=0.0007)
