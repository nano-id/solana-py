"""Paper simulation of delta-neutral funding harvest on historical rates."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pairs_trading.funding.fees import FeeConfig, round_trip_cost
from pairs_trading.funding.risk import harvest_side


@dataclass(frozen=True)
class PaperResult:
    symbol: str
    side: str
    n_settlements: int
    notional_usdt: float
    funding_pnl: float
    fee_pnl: float
    basis_pnl: float
    net_pnl: float
    net_return: float
    ann_return: float
    hold_days: float


def simulate_funding_harvest(
    rates: pd.Series,
    *,
    symbol: str,
    notional_usdt: float,
    fee_config: FeeConfig,
    settlements_per_day: float = 3.0,
    side_rate: float | None = None,
    entry_basis: float = 0.0,
    exit_basis: float = 0.0,
) -> PaperResult:
    s = rates.astype(float).dropna()
    if s.empty or notional_usdt <= 0:
        return PaperResult(
            symbol=symbol,
            side="flat",
            n_settlements=0,
            notional_usdt=float(notional_usdt),
            funding_pnl=0.0,
            fee_pnl=0.0,
            basis_pnl=0.0,
            net_pnl=0.0,
            net_return=0.0,
            ann_return=0.0,
            hold_days=0.0,
        )

    signal = float(s.mean()) if side_rate is None else float(side_rate)
    side = harvest_side(signal)
    sign = 1.0 if side == "short_perp_long_spot" else -1.0
    funding_pnl = float(sign * (s.to_numpy() * notional_usdt).sum())
    fees = round_trip_cost(fee_config) * notional_usdt
    basis_pnl = float(-sign * (exit_basis - entry_basis) * notional_usdt)
    net = funding_pnl - fees + basis_pnl
    n = len(s)
    hold_days = n / float(settlements_per_day)
    net_ret = net / notional_usdt
    ann = net_ret * (365.0 / hold_days) if hold_days > 0 else 0.0
    return PaperResult(
        symbol=symbol,
        side=side,
        n_settlements=n,
        notional_usdt=float(notional_usdt),
        funding_pnl=funding_pnl,
        fee_pnl=-fees,
        basis_pnl=basis_pnl,
        net_pnl=net,
        net_return=net_ret,
        ann_return=ann,
        hold_days=hold_days,
    )


def paper_results_frame(results: list[PaperResult]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "symbol": r.symbol,
                "side": r.side,
                "n_settlements": r.n_settlements,
                "notional_usdt": r.notional_usdt,
                "funding_pnl": r.funding_pnl,
                "fee_pnl": r.fee_pnl,
                "basis_pnl": r.basis_pnl,
                "net_pnl": r.net_pnl,
                "net_return": r.net_return,
                "ann_return": r.ann_return,
                "hold_days": r.hold_days,
                "positive": r.net_pnl > 0,
            }
            for r in results
        ]
    )
