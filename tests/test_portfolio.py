"""Tests for single-wallet portfolio router."""

from __future__ import annotations

import numpy as np

from market_making.config import MarketMakingConfig, load_mm_config
from market_making.portfolio import run_portfolio_panel


def _synthetic_panel(n: int = 60, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "AAA/USDT": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
        "BBB/USDT": 50 * np.exp(np.cumsum(rng.normal(0, 0.012, n))),
        "CCC/USDT": 20 * np.exp(np.cumsum(rng.normal(0, 0.015, n))),
    }


def test_portfolio_lock_invariant() -> None:
    cfg = MarketMakingConfig(commission_pct=0.07, symbols=["AAA/USDT", "BBB/USDT", "CCC/USDT"])
    result, frame = run_portfolio_panel(cfg, _synthetic_panel(), seed=7, initial_equity=1000.0)
    assert result.n_entries >= 1
    cur = None
    for _, row in frame.iterrows():
        if row["action"] == "enter":
            assert cur is None
            cur = row["active_symbol"]
        elif row["action"] == "close":
            assert cur is not None
            cur = None
        elif row["action"] == "manage":
            assert row["active_symbol"] == cur
    assert result.interpretation_allowed is False


def test_portfolio_picks_single_when_multiple() -> None:
    cfg = MarketMakingConfig(commission_pct=0.07)
    # High fill prob so multi-signal bars are likely
    cfg.sim.fill_prob = 0.95
    result, frame = run_portfolio_panel(
        cfg, _synthetic_panel(n=80, seed=1), seed=3, initial_equity=1000.0
    )
    # Enter rows never have n_candidates implying multiple simultaneous actives
    enters = frame.loc[frame["action"] == "enter"]
    assert (enters["n_candidates"] >= 1).all()
    # At most one active symbol on any manage/enter bar
    busy = frame.loc[frame["action"].isin(["enter", "manage"])]
    assert busy["active_symbol"].notna().all()


def test_config_universe_has_uni_not_trx() -> None:
    cfg = load_mm_config("config/market_making.yaml")
    assert "UNI/USDT" in cfg.symbols
    assert "TRX/USDT" not in cfg.symbols
    assert len(cfg.symbols) == 10
