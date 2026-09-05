"""Quote engine: AS reservation/spread + fee positivity gate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from market_making.as_math import (
    as_half_spread_bps,
    ewma_vol_per_bar,
    fee_adverse_floor_bps,
    inventory_size_multipliers,
    reservation_price,
)
from market_making.config import MarketMakingConfig
from market_making.fees import is_positive_half_spread, round_trip_net_edge_bps
from market_making.inventory import InventoryState


@dataclass(frozen=True)
class Quote:
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    mid: float
    reservation: float
    half_spread_bps: float
    net_edge_bps_if_matched: float
    quoting: bool
    reason: str
    sigma_bps: float = 0.0


class QuoteEngine:
    def __init__(self, cfg: MarketMakingConfig) -> None:
        self.cfg = cfg

    def _sigma(self, mid_history: np.ndarray) -> float:
        hist = np.asarray(mid_history, dtype=float)
        hist = hist[np.isfinite(hist) & (hist > 0)]
        if len(hist) < 2:
            return 0.0
        w = hist[-self.cfg.vol_lookback :]
        lr = np.diff(np.log(w))
        return ewma_vol_per_bar(lr, halflife=self.cfg.ewma_halflife)

    def half_spread_bps(
        self, mid_history: np.ndarray
    ) -> tuple[float, float]:
        sigma = self._sigma(mid_history)
        sigma_bps = sigma * 1e4
        floor = fee_adverse_floor_bps(
            commission_bps=self.cfg.commission_bps,
            edge_buffer_bps=self.cfg.edge_buffer_bps,
            min_half_spread_bps=self.cfg.min_half_spread_bps,
            sigma_bps=sigma_bps,
            adverse_vol_mult=self.cfg.adverse_vol_mult,
        )
        vol_floor = self.cfg.vol_spread_mult * sigma_bps
        if self.cfg.use_as_quotes:
            as_half = as_half_spread_bps(
                sigma,
                gamma=self.cfg.gamma,
                kappa=self.cfg.kappa,
                tau=self.cfg.tau_bars,
                liq_bps=self.cfg.liq_bps,
            )
            half = max(floor, vol_floor, as_half)
        else:
            half = max(self.cfg.required_half_spread_bps, vol_floor)
        return float(half), float(sigma_bps)

    def quote(
        self,
        mid: float,
        inv: InventoryState,
        *,
        mid_history: np.ndarray | None = None,
    ) -> Quote:
        hist = mid_history if mid_history is not None else np.asarray([mid], dtype=float)
        if not np.isfinite(mid) or mid <= 0:
            return self._dead(mid, reason="invalid_mid")
        if inv.halted:
            return self._dead(mid, reason="inventory_halt")

        q_norm = float(np.clip(inv.deviation / self.cfg.max_inventory, -1.0, 1.0))
        half_bps, sigma_bps = self.half_spread_bps(hist)
        if not is_positive_half_spread(half_bps, self.cfg):
            return self._dead(
                mid,
                reason="positivity_gate",
                half_bps=half_bps,
                edge=round_trip_net_edge_bps(half_bps, self.cfg),
                sigma_bps=sigma_bps,
            )

        sigma = sigma_bps / 1e4
        if self.cfg.use_as_quotes:
            reservation = reservation_price(
                mid, q_norm, sigma, gamma=self.cfg.gamma, tau=self.cfg.tau_bars
            )
        else:
            skew = (self.cfg.inventory_skew_bps / 1e4) * q_norm
            reservation = mid * (1.0 - skew)

        half = half_bps / 1e4
        bid = reservation * (1.0 - half)
        ask = reservation * (1.0 + half)
        bid = min(bid, mid * (1.0 - 0.5 * half))
        ask = max(ask, mid * (1.0 + 0.5 * half))
        if bid <= 0 or ask <= bid:
            return self._dead(mid, reason="crossed_book", half_bps=half_bps, sigma_bps=sigma_bps)

        base_size = self.cfg.max_inventory * self.cfg.size_frac_of_max
        if self.cfg.use_as_quotes:
            bid_mult, ask_mult = inventory_size_multipliers(q_norm, self.cfg.size_eta)
        else:
            bid_mult = max(0.05, 1.0 - max(0.0, q_norm))
            ask_mult = max(0.05, 1.0 - max(0.0, -q_norm))
        bid_size = base_size * max(0.05, bid_mult)
        ask_size = base_size * max(0.05, ask_mult)
        room_buy = self.cfg.max_inventory - inv.base
        room_sell = inv.base + self.cfg.max_inventory
        bid_size = float(max(0.0, min(bid_size, room_buy)))
        ask_size = float(max(0.0, min(ask_size, room_sell)))
        quoting = bid_size > 0 or ask_size > 0
        return Quote(
            bid=float(bid),
            ask=float(ask),
            bid_size=bid_size,
            ask_size=ask_size,
            mid=float(mid),
            reservation=float(reservation),
            half_spread_bps=float(half_bps),
            net_edge_bps_if_matched=float(round_trip_net_edge_bps(half_bps, self.cfg)),
            quoting=quoting,
            reason="ok" if quoting else "no_size",
            sigma_bps=float(sigma_bps),
        )

    def _dead(
        self,
        mid: float,
        *,
        reason: str,
        half_bps: float = 0.0,
        edge: float = 0.0,
        sigma_bps: float = 0.0,
    ) -> Quote:
        return Quote(
            bid=0.0,
            ask=0.0,
            bid_size=0.0,
            ask_size=0.0,
            mid=float(mid) if np.isfinite(mid) else 0.0,
            reservation=float(mid) if np.isfinite(mid) else 0.0,
            half_spread_bps=float(half_bps),
            net_edge_bps_if_matched=float(edge),
            quoting=False,
            reason=reason,
            sigma_bps=float(sigma_bps),
        )
