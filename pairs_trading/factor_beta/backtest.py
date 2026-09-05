"""Residual z-score paper PnL — gross vs net with OKX-scale RT costs.

Position in residual space:
  long_spread  → +1 (expect ε to rise toward 0)
  short_spread → −1
Daily log-PnL ≈ position_{t-1} * (ε_t − ε_{t-1}); reported in bps (*1e4).
Each completed round-trip charges ``round_trip_bps`` (Rule 5a).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pairs_trading.signal.engine import Side, SignalConfig, generate_signals


@dataclass(frozen=True)
class TradePnL:
    entry_bar: int
    exit_bar: int
    side: str
    gross_bps: float
    cost_bps: float
    net_bps: float
    exit_reason: str


@dataclass(frozen=True)
class BacktestResult:
    n_bars: int
    n_trades: int
    gross_bps: float
    cost_bps: float
    net_bps: float
    mean_gross_per_trade_bps: float
    mean_net_per_trade_bps: float
    cost_leakage_ratio: float  # cost / |gross| if gross≠0
    bars_in_position: int
    trades: tuple[TradePnL, ...]


def _side_sign(side: str) -> float:
    if side == Side.LONG_SPREAD.value:
        return 1.0
    if side == Side.SHORT_SPREAD.value:
        return -1.0
    return 0.0


def simulate_residual_pnl(
    residual: pd.Series,
    signal_cfg: SignalConfig,
    *,
    round_trip_bps: float,
    start: int | None = None,
    end: int | None = None,
) -> BacktestResult:
    """Causal z-signals on full series; attribute PnL to [start, end).

    Positions may be inherited from before ``start``; entry cost is charged
    only when the entry event bar is inside the attribution window (or at
    the first bar if a position is already open at ``start`` — marked as
    carry-in without charging a new RT until the next exit/re-entry).
    Carry-in: no entry fee at window start; exit inside window still pays RT
    (conservative for incomplete RTs opened earlier).
    """
    residual = residual.astype(float)
    frame, events = generate_signals(residual, signal_cfg)
    n = len(frame)
    start_i = 0 if start is None else max(0, int(start))
    end_i = n if end is None else min(n, int(end))
    if end_i <= start_i:
        return BacktestResult(
            n_bars=0,
            n_trades=0,
            gross_bps=0.0,
            cost_bps=0.0,
            net_bps=0.0,
            mean_gross_per_trade_bps=0.0,
            mean_net_per_trade_bps=0.0,
            cost_leakage_ratio=float("nan"),
            bars_in_position=0,
            trades=(),
        )

    spreads = residual.to_numpy(dtype=float)
    sides = frame["side"].to_numpy()
    reasons = frame["reason"].to_numpy()

    trades: list[TradePnL] = []
    open_side: str | None = None
    open_bar: int | None = None
    open_gross = 0.0
    charged_entry = False

    # Seed open position from bar before window if any
    if start_i > 0 and sides[start_i - 1] != Side.FLAT.value:
        open_side = str(sides[start_i - 1])
        open_bar = start_i - 1
        open_gross = 0.0
        charged_entry = False  # carry-in

    window_gross = 0.0
    window_cost = 0.0
    bars_in_pos = 0

    for i in range(start_i, end_i):
        if sides[i] != Side.FLAT.value:
            bars_in_pos += 1
        # Mark-to-market from previous bar
        if i > 0 and open_side is not None:
            d_eps = float(spreads[i] - spreads[i - 1])
            if np.isfinite(d_eps):
                day = _side_sign(open_side) * d_eps * 1e4
                open_gross += day
                window_gross += day

        reason = str(reasons[i])
        if reason in ("enter_long", "enter_short"):
            open_side = str(sides[i])
            open_bar = i
            open_gross = 0.0
            charged_entry = True
        elif reason in ("exit_mean", "exit_stop") and open_side is not None:
            cost = float(round_trip_bps) if charged_entry else float(round_trip_bps)
            # Always charge one RT on exit of a attributed trade.
            # Carry-in exits: still charge RT (closing the book); note in reason.
            exit_reason = reason if charged_entry else f"{reason}_carry_in"
            net = open_gross - cost
            trades.append(
                TradePnL(
                    entry_bar=int(open_bar if open_bar is not None else i),
                    exit_bar=i,
                    side=open_side,
                    gross_bps=float(open_gross),
                    cost_bps=cost,
                    net_bps=float(net),
                    exit_reason=exit_reason,
                )
            )
            window_cost += cost
            open_side = None
            open_bar = None
            open_gross = 0.0
            charged_entry = False

    # Flat open position at window end — unrealized NOT counted as trade;
    # MTM already in window_gross. No exit fee until closed.
    n_trades = len(trades)
    mean_g = float(np.mean([t.gross_bps for t in trades])) if n_trades else 0.0
    mean_n = float(np.mean([t.net_bps for t in trades])) if n_trades else 0.0
    net = window_gross - window_cost
    leak = (
        float(window_cost / abs(window_gross))
        if abs(window_gross) > 1e-12
        else float("nan")
    )
    return BacktestResult(
        n_bars=int(end_i - start_i),
        n_trades=n_trades,
        gross_bps=float(window_gross),
        cost_bps=float(window_cost),
        net_bps=float(net),
        mean_gross_per_trade_bps=mean_g,
        mean_net_per_trade_bps=mean_n,
        cost_leakage_ratio=leak,
        bars_in_position=int(bars_in_pos),
        trades=tuple(trades),
    )
