"""Avellaneda–Stoikov style math with well-posed normalized units.

Inventory is normalized q ∈ [-1, 1]. Volatility σ is per-bar log-return std;
σ_bps = 1e4 · σ.

Reservation mid (inventory risk in return space):
  skew_frac = γ · q · σ_bps · √τ / 1e4
  r = mid · (1 − clip(skew_frac, ±5%))

Indifference half-spread (bps). Classic (1/γ)ln(1+γ/κ) as a *price fraction*
explodes for γ~0.1; we keep the AS structure but place the liquidity term in bps:

  half_bps_AS = γ · σ_bps · √τ + λ_liq · ln(1 + γ/κ)

Fee / adverse floor (always):
  half_bps ≥ max(min_half, commission_bps + edge_buffer + λ_adv · σ_bps)

Final:
  half_bps = max(half_bps_AS, fee_adverse_floor, vol_floor)

Size tilt:
  bid_mult = exp(−η · max(0, q))
  ask_mult = exp(−η · max(0, −q))
"""

from __future__ import annotations

import math

import numpy as np


def ewma_vol_per_bar(log_returns: np.ndarray, *, halflife: float) -> float:
    """EWMA std of log-returns (per-bar σ)."""
    x = np.asarray(log_returns, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0
    if len(x) == 1:
        return float(abs(x[0]))
    hl = max(float(halflife), 1.0)
    alpha = 1.0 - math.exp(math.log(0.5) / hl)
    var = float(x[0] * x[0])
    for r in x[1:]:
        var = (1.0 - alpha) * var + alpha * float(r * r)
    return math.sqrt(max(var, 0.0))


def reservation_price(
    mid: float,
    q_norm: float,
    sigma: float,
    *,
    gamma: float,
    tau: float,
) -> float:
    """Inventory-adjusted reservation mid."""
    sigma_bps = float(sigma) * 1e4
    skew_frac = gamma * q_norm * sigma_bps * math.sqrt(max(tau, 0.0)) / 1e4
    skew_frac = float(np.clip(skew_frac, -0.05, 0.05))
    return float(mid * (1.0 - skew_frac))


def as_half_spread_bps(
    sigma: float,
    *,
    gamma: float,
    kappa: float,
    tau: float,
    liq_bps: float = 10.0,
) -> float:
    """Normalized AS half-spread in bps (see module docstring)."""
    gamma = max(float(gamma), 1e-12)
    kappa = max(float(kappa), 1e-12)
    sigma_bps = max(float(sigma) * 1e4, 0.0)
    vol_term = gamma * sigma_bps * math.sqrt(max(tau, 0.0))
    liq_term = float(liq_bps) * math.log(1.0 + gamma / kappa)
    return float(max(vol_term + liq_term, 0.0))


def fee_adverse_floor_bps(
    *,
    commission_bps: float,
    edge_buffer_bps: float,
    min_half_spread_bps: float,
    sigma_bps: float,
    adverse_vol_mult: float,
) -> float:
    return float(
        max(
            min_half_spread_bps,
            commission_bps + edge_buffer_bps + adverse_vol_mult * max(sigma_bps, 0.0),
        )
    )


def inventory_size_multipliers(q_norm: float, eta: float) -> tuple[float, float]:
    eta = max(float(eta), 0.0)
    q = float(np.clip(q_norm, -1.0, 1.0))
    return math.exp(-eta * max(0.0, q)), math.exp(-eta * max(0.0, -q))
