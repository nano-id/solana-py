"""Hourly calibration smoke tests."""

from __future__ import annotations

from market_making.calibrate_hourly import (
    calibrate_hourly,
    default_1h_param_grid,
    fifo_hold_hours_from_fills,
)
from market_making.config import MarketMakingConfig
import pandas as pd


def test_default_grid_nonempty() -> None:
    g = default_1h_param_grid()
    assert len(g) >= 20
    assert all("tau_bars" in p and p["tau_bars"] >= 16 for p in g)


def test_fifo_hold_hours() -> None:
    frame = pd.DataFrame(
        [
            {"bar": 0, "buy_fill": True, "buy_size": 1.0, "sell_fill": False, "sell_size": 0.0},
            {"bar": 3, "buy_fill": False, "buy_size": 0.0, "sell_fill": True, "sell_size": 1.0},
        ]
    )
    h = fifo_hold_hours_from_fills(frame, hours_per_bar=1.0)
    assert h["n_holds"] == 1
    assert h["mean_h"] == 3.0


def test_calibrate_hourly_tiny_grid() -> None:
    cfg = MarketMakingConfig(
        symbols=["BTC/USDT", "ETH/USDT"],
        commission_pct=0.07,
        storage_root="data",
        exchange_id="okx",
    )
    grid = [
        {
            "gamma": 0.12,
            "kappa": 0.8,
            "tau_bars": 32.0,
            "size_eta": 3.5,
            "adverse_vol_mult": 0.45,
            "vol_lookback": 96.0,
            "ewma_halflife": 48.0,
            "liq_bps": 6.0,
            "vol_spread_mult": 0.25,
            "inventory_halt_frac": 0.80,
        },
        {
            "gamma": 0.08,
            "kappa": 1.2,
            "tau_bars": 16.0,
            "size_eta": 2.5,
            "adverse_vol_mult": 0.35,
            "vol_lookback": 96.0,
            "ewma_halflife": 48.0,
            "liq_bps": 6.0,
            "vol_spread_mult": 0.25,
            "inventory_halt_frac": 0.80,
        },
    ]
    result = calibrate_hourly(
        cfg,
        is_months=[(2025, 2)],
        oos_months=[(2025, 3)],
        timeframe="1h",
        seed=1,
        grid=grid,
        top_k=2,
        portfolio_equity=1000.0,
    )
    assert result.interpretation_allowed is False
    assert "gamma" in result.selected_params
    assert "2025-03" in result.portfolio_oos
