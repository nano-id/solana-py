"""Quote engine: reservation price + positivity-gated half-spread."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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


def _vol_bps(mids: np.ndarray, lookback: int) -> float:
    if len(mids) < 2:
        return 0.0
    w = np.asarray(mids[-lookback:], dtype=float)
    if len(w) < 2:
        return 0.0
    lr = np.diff(np.log(w))
    if len(lr) == 0:
        return 0.0
    return float(np.std(lr, ddof=1) * 1e4) if len(lr) > 1 else float(abs(lr[0]) * 1e4)


class QuoteEngine:
    def __init__(self, cfg: MarketMakingConfig) -> None:
        self.cfg = cfg

    def half_spread_bps(self, mid_history: np.ndarray) -> float:
        vol = _vol_bps(mid_history, self.cfg.vol_lookback)
        return float(
            max(self.cfg.required_half_spread_bps, self.cfg.vol_spread_mult * vol)
        )

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

        half_bps = self.half_spread_bps(hist)
        if not is_positive_half_spread(half_bps, self.cfg):
            return self._dead(
                mid,
                reason="positivity_gate",
                half_bps=half_bps,
                edge=round_trip_net_edge_bps(half_bps, self.cfg),
            )

        norm = inv.deviation / self.cfg.max_inventory
        skew = (self.cfg.inventory_skew_bps / 1e4) * norm
        reservation = mid * (1.0 - skew)
        half = half_bps / 1e4
        bid = reservation * (1.0 - half)
        ask = reservation * (1.0 + half)
        bid = min(bid, mid * (1.0 - 0.5 * half))
        ask = max(ask, mid * (1.0 + 0.5 * half))
        if bid <= 0 or ask <= bid:
            return self._dead(mid, reason="crossed_book", half_bps=half_bps)

        base_size = self.cfg.max_inventory * self.cfg.size_frac_of_max
        bid_size = base_size * max(0.05, 1.0 - max(0.0, norm))
        ask_size = base_size * max(0.05, 1.0 - max(0.0, -norm))
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
        )

    def _dead(
        self,
        mid: float,
        *,
        reason: str,
        half_bps: float = 0.0,
        edge: float = 0.0,
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
        )
