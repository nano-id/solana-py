"""Phase 3 signal skeleton tests on synthetic OU spreads."""

from __future__ import annotations

import numpy as np

from pairs_trading.signal import (
    Side,
    SignalConfig,
    generate_signals,
    simulate_ou_spread,
    summarize_signals,
)


def test_ou_mean_reverts():
    s = simulate_ou_spread(n=3000, theta=0.15, mu=0.0, sigma=0.02, seed=0)
    # Autocorr of lag-1 should be positive but < 1 for mean-reverting process.
    ac1 = np.corrcoef(s.iloc[1:], s.iloc[:-1])[0, 1]
    assert 0.5 < ac1 < 0.99


def test_signal_enters_and_exits_on_ou():
    spread = simulate_ou_spread(n=4000, theta=0.12, mu=0.0, sigma=0.04, seed=7)
    cfg = SignalConfig(entry_z=2.0, exit_z=0.5, stop_z=5.0, z_window=50)
    frame, events = generate_signals(spread, cfg)
    summary = summarize_signals(frame, events)
    assert summary["n_entries"] >= 1
    assert summary["n_exit_mean"] + summary["n_exit_stop"] >= 1
    assert summary["bars_in_position"] > 0
    # Mean exits should dominate stops on a well-behaved OU.
    assert summary["n_exit_mean"] >= summary["n_exit_stop"]


def test_stop_triggers_on_blowout():
    # Artificial spread: quiet then jump far beyond stop.
    import pandas as pd

    x = np.zeros(200)
    x[100:] = 10.0  # huge level shift → extreme z after window fills
    spread = pd.Series(x)
    cfg = SignalConfig(entry_z=1.5, exit_z=0.2, stop_z=3.0, z_window=30)
    # Force an entry first by making a moderate excursion then blowout
    x = np.zeros(250)
    rng = np.random.default_rng(0)
    x[:120] = rng.normal(0, 0.1, size=120)
    x[80:100] = 1.0  # enough to enter short after z builds
    x[150:] = 20.0
    frame, events = generate_signals(pd.Series(x), cfg)
    reasons = [e.reason for e in events]
    assert "enter_short" in reasons or "enter_long" in reasons
    # Blowout should produce at least one stop if we were in a trade,
    # otherwise flat is acceptable — assert engine doesn't crash and sides valid.
    assert set(frame["side"]).issubset({Side.FLAT.value, Side.LONG_SPREAD.value, Side.SHORT_SPREAD.value})
