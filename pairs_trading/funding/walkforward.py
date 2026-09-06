"""Blind walk-forward funding harvest (no look-ahead side selection).

Protocol
--------
At each fold boundary ``t``:
  - train = rates[t - train_n : t]   # past only
  - side  = sign(mean(train))       # decided without future
  - optional gate: |mean(train)| and train persistence must clear fee drag
  - test  = rates[t : t + test_n]   # future only → PnL
  - fee   = one round-trip charged when entering / flipping side

This is the OOS counterpart to the in-sample ``simulate_funding_harvest`` path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pairs_trading.funding.config import WalkForwardConfig
from pairs_trading.funding.fees import FeeConfig, round_trip_cost
from pairs_trading.funding.persistence import funding_persistence
from pairs_trading.funding.risk import harvest_side

if TYPE_CHECKING:
    from pairs_trading.funding.config import FundingAppConfig


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    side: str
    train_mean: float
    train_efficiency: float
    traded: bool
    skip_reason: str
    n_test: int
    funding_pnl: float
    fee_pnl: float
    net_pnl: float


@dataclass(frozen=True)
class WalkForwardResult:
    symbol: str
    notional_usdt: float
    n_folds: int
    n_traded_folds: int
    n_test_settlements: int
    funding_pnl: float
    fee_pnl: float
    net_pnl: float
    net_return: float
    ann_return: float
    hold_days: float
    fold_hit_rate: float
    folds: tuple[WalkForwardFold, ...]


def _side_sign(side: str) -> float:
    return 1.0 if side == "short_perp_long_spot" else -1.0


def _train_passes_gate(
    train: pd.Series,
    *,
    cfg: WalkForwardConfig,
    fee_config: FeeConfig,
) -> tuple[bool, str, float, float]:
    """Blind entry gate using only train stats.

    Compare expected funding collected over the upcoming *test* window
    to one round-trip fee (not a 7-day-annualized fee drag, which is
    pathologically harsh for weekly folds).
    """
    mean_rate = float(train.mean())
    pers = funding_persistence(train)
    eff = pers.get("harvest_efficiency", float("nan"))
    if eff != eff:
        eff = 1.0
    if abs(mean_rate) < cfg.min_abs_train_mean:
        return False, "below_noise", mean_rate, float(eff)
    if not cfg.require_train_edge:
        return True, "ok", mean_rate, float(eff)
    expected_collect = abs(mean_rate) * float(eff) * float(cfg.test_settlements)
    fee = round_trip_cost(fee_config)
    if expected_collect <= fee:
        return False, "no_edge_vs_fees", mean_rate, float(eff)
    return True, "ok", mean_rate, float(eff)


def walk_forward_funding_harvest(
    rates: pd.Series,
    *,
    symbol: str,
    notional_usdt: float,
    fee_config: FeeConfig,
    cfg: WalkForwardConfig | None = None,
) -> WalkForwardResult:
    """Rolling train→test funding harvest with side chosen only from train."""
    cfg = cfg or WalkForwardConfig()
    s = rates.astype(float).dropna().reset_index(drop=True)
    empty = WalkForwardResult(
        symbol=symbol,
        notional_usdt=float(notional_usdt),
        n_folds=0,
        n_traded_folds=0,
        n_test_settlements=0,
        funding_pnl=0.0,
        fee_pnl=0.0,
        net_pnl=0.0,
        net_return=0.0,
        ann_return=0.0,
        hold_days=0.0,
        fold_hit_rate=float("nan"),
        folds=(),
    )
    if s.empty or notional_usdt <= 0:
        return empty

    n = len(s)
    need = cfg.train_settlements + cfg.test_settlements
    if n < need:
        return empty

    folds: list[WalkForwardFold] = []
    prev_side: str | None = None
    fold_i = 0
    t = cfg.train_settlements
    while t + cfg.test_settlements <= n:
        train = s.iloc[t - cfg.train_settlements : t]
        test = s.iloc[t : t + cfg.test_settlements]
        ok, reason, mean_rate, eff = _train_passes_gate(train, cfg=cfg, fee_config=fee_config)
        side = harvest_side(mean_rate)

        if not ok:
            folds.append(
                WalkForwardFold(
                    fold=fold_i,
                    train_start=t - cfg.train_settlements,
                    train_end=t,
                    test_start=t,
                    test_end=t + cfg.test_settlements,
                    side="flat",
                    train_mean=mean_rate,
                    train_efficiency=eff,
                    traded=False,
                    skip_reason=reason,
                    n_test=0,
                    funding_pnl=0.0,
                    fee_pnl=0.0,
                    net_pnl=0.0,
                )
            )
        else:
            sign = _side_sign(side)
            funding_pnl = float(sign * (test.to_numpy() * notional_usdt).sum())
            fee = 0.0
            if cfg.charge_fees_on_rebalance and (prev_side is None or prev_side != side):
                fee = round_trip_cost(fee_config) * notional_usdt
            net = funding_pnl - fee
            folds.append(
                WalkForwardFold(
                    fold=fold_i,
                    train_start=t - cfg.train_settlements,
                    train_end=t,
                    test_start=t,
                    test_end=t + cfg.test_settlements,
                    side=side,
                    train_mean=mean_rate,
                    train_efficiency=eff,
                    traded=True,
                    skip_reason="ok",
                    n_test=len(test),
                    funding_pnl=funding_pnl,
                    fee_pnl=-fee,
                    net_pnl=net,
                )
            )
            prev_side = side

        fold_i += 1
        t += cfg.step_settlements

    if not folds:
        return empty

    traded = [f for f in folds if f.traded]
    # Close the book once after the last traded fold (still causal: fee after OOS pnl).
    if cfg.charge_exit_fee and traded and cfg.charge_fees_on_rebalance:
        last = traded[-1]
        exit_fee = round_trip_cost(fee_config) * notional_usdt
        traded[-1] = WalkForwardFold(
            fold=last.fold,
            train_start=last.train_start,
            train_end=last.train_end,
            test_start=last.test_start,
            test_end=last.test_end,
            side=last.side,
            train_mean=last.train_mean,
            train_efficiency=last.train_efficiency,
            traded=True,
            skip_reason=last.skip_reason,
            n_test=last.n_test,
            funding_pnl=last.funding_pnl,
            fee_pnl=last.fee_pnl - exit_fee,
            net_pnl=last.net_pnl - exit_fee,
        )
        # also update folds tuple entry
        folds = list(folds)
        for i, f in enumerate(folds):
            if f.fold == last.fold and f.traded:
                folds[i] = traded[-1]
                break
    funding = float(sum(f.funding_pnl for f in traded))
    fees = float(sum(f.fee_pnl for f in traded))
    net = float(sum(f.net_pnl for f in traded))
    n_test = int(sum(f.n_test for f in traded))
    hold_days = n_test / float(cfg.settlements_per_day) if n_test else 0.0
    net_ret = net / notional_usdt if notional_usdt else 0.0
    ann = net_ret * (365.0 / hold_days) if hold_days > 0 else 0.0
    hits = [f.net_pnl > 0 for f in traded]
    hit_rate = float(np.mean(hits)) if hits else float("nan")

    return WalkForwardResult(
        symbol=symbol,
        notional_usdt=float(notional_usdt),
        n_folds=len(folds),
        n_traded_folds=len(traded),
        n_test_settlements=n_test,
        funding_pnl=funding,
        fee_pnl=fees,
        net_pnl=net,
        net_return=net_ret,
        ann_return=ann,
        hold_days=hold_days,
        fold_hit_rate=hit_rate,
        folds=tuple(folds),
    )


def walk_forward_results_frame(results: list[WalkForwardResult]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "symbol": r.symbol,
                "notional_usdt": r.notional_usdt,
                "n_folds": r.n_folds,
                "n_traded_folds": r.n_traded_folds,
                "n_test_settlements": r.n_test_settlements,
                "funding_pnl": r.funding_pnl,
                "fee_pnl": r.fee_pnl,
                "net_pnl": r.net_pnl,
                "net_return": r.net_return,
                "ann_return": r.ann_return,
                "hold_days": r.hold_days,
                "fold_hit_rate": r.fold_hit_rate,
                "positive": r.net_pnl > 0,
            }
            for r in results
        ]
    )


def walk_forward_folds_frame(result: WalkForwardResult) -> pd.DataFrame:
    if not result.folds:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "symbol": result.symbol,
                "fold": f.fold,
                "train_start": f.train_start,
                "train_end": f.train_end,
                "test_start": f.test_start,
                "test_end": f.test_end,
                "side": f.side,
                "train_mean": f.train_mean,
                "train_efficiency": f.train_efficiency,
                "traded": f.traded,
                "skip_reason": f.skip_reason,
                "n_test": f.n_test,
                "funding_pnl": f.funding_pnl,
                "fee_pnl": f.fee_pnl,
                "net_pnl": f.net_pnl,
            }
            for f in result.folds
        ]
    )


def run_walkforward_universe(
    config: FundingAppConfig | None = None,
    *,
    exchange_id: str | None = None,
    symbols: list[str] | None = None,
    wf_config: WalkForwardConfig | None = None,
    fee_config: FeeConfig | None = None,
    notional_usdt: float | None = None,
) -> tuple[pd.DataFrame, list[WalkForwardResult]]:
    """Blind walk-forward across stored funding symbols."""
    from pairs_trading.funding.config import load_funding_config
    from pairs_trading.funding.harvest import list_stored_swap_symbols
    from pairs_trading.funding.storage import FundingStore

    config = config or load_funding_config()
    fee_config = fee_config or config.fees
    risk = config.risk
    wf_config = wf_config or config.walkforward
    wf_config = wf_config.model_copy(
        update={"settlements_per_day": config.funding.settlements_per_day}
    )
    eid = exchange_id or config.exchange.id
    store = FundingStore(config.storage.root, eid)
    if symbols:
        available = [s for s in symbols if store.exists(s)]
    else:
        available = list_stored_swap_symbols(store)
        if not available:
            available = [s for s in config.funding.default_symbols if store.exists(s)]

    default_notional = (
        float(notional_usdt)
        if notional_usdt is not None
        else float(risk.equity_usdt * risk.target_gross_fraction)
    )
    default_notional = min(default_notional, float(risk.equity_usdt * risk.max_name_fraction))

    results: list[WalkForwardResult] = []
    for sym in available:
        hist = store.load(sym)
        if hist.empty or "funding_rate" not in hist.columns:
            continue
        # Fixed risk notional — do not size from full-sample mean (look-ahead).
        results.append(
            walk_forward_funding_harvest(
                hist["funding_rate"],
                symbol=sym,
                notional_usdt=default_notional,
                fee_config=fee_config,
                cfg=wf_config,
            )
        )
    frame = walk_forward_results_frame(results)
    if not frame.empty:
        frame = frame.sort_values("net_pnl", ascending=False).reset_index(drop=True)
        frame.insert(0, "rank", range(1, len(frame) + 1))
    return frame, results
