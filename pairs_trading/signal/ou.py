"""Ornstein-Uhlenbeck synthetic spread generator."""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_ou_spread(
    n: int = 2000,
    *,
    theta: float = 0.1,
    mu: float = 0.0,
    sigma: float = 0.05,
    x0: float = 0.0,
    seed: int = 42,
) -> pd.Series:
    """Discrete OU: x_{t+1} = x_t + θ(μ − x_t) + σ ε_t."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    x[0] = x0
    for t in range(1, n):
        x[t] = x[t - 1] + theta * (mu - x[t - 1]) + sigma * rng.normal()
    return pd.Series(x, name="spread")
