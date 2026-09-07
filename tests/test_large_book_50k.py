"""Smoke test for $50k large-book research module."""

from __future__ import annotations

from market_making.large_book_50k import LARGE_UNIVERSE, base_50k_config, large_param_grid


def test_base_50k_profile() -> None:
    cfg = base_50k_config()
    assert cfg.symbols == LARGE_UNIVERSE
    assert cfg.portfolio_initial_equity == 50_000.0
    assert cfg.size_frac_of_max == 0.05
    assert cfg.sim.fill_prob == 0.20
    assert cfg.interpretation_allowed is False


def test_large_grid_nonempty() -> None:
    g = large_param_grid()
    assert len(g) >= 20
    assert all("gamma" in p and "tau_bars" in p for p in g)
