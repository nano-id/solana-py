from __future__ import annotations

import pandas as pd

from pairs_trading.funding.config import WalkForwardConfig
from pairs_trading.funding.fees import FeeConfig
from pairs_trading.funding.walkforward import walk_forward_funding_harvest


def test_walkforward_no_lookahead_on_persistent_positive() -> None:
    train = [0.0002] * 90
    test = [0.0002] * 90
    rates = pd.Series(train + test)
    res = walk_forward_funding_harvest(
        rates,
        symbol="SYN/USDT:USDT",
        notional_usdt=1000.0,
        fee_config=FeeConfig(),
        cfg=WalkForwardConfig(
            train_settlements=90,
            test_settlements=90,
            step_settlements=90,
            require_train_edge=True,
            charge_exit_fee=True,
        ),
    )
    assert res.n_traded_folds == 1
    assert res.net_pnl > 0
    assert res.folds[0].side == "short_perp_long_spot"


def test_walkforward_side_uses_train_not_full_sample() -> None:
    # Full-sample mean is negative; train mean is positive → side still short.
    train = [0.0002] * 90
    test = [-0.001] * 90
    rates = pd.Series(train + test)
    assert float(rates.mean()) < 0
    res = walk_forward_funding_harvest(
        rates,
        symbol="TRAP/USDT:USDT",
        notional_usdt=1000.0,
        fee_config=FeeConfig(),
        cfg=WalkForwardConfig(
            train_settlements=90,
            test_settlements=90,
            step_settlements=90,
            require_train_edge=False,
            charge_exit_fee=False,
            charge_fees_on_rebalance=False,
        ),
    )
    assert res.n_traded_folds == 1
    assert res.folds[0].side == "short_perp_long_spot"
    assert res.funding_pnl < 0  # wrong-side OOS


def test_walkforward_skips_when_no_edge_vs_fees() -> None:
    # Above noise floor but cannot clear round-trip over 90 settlements.
    # 2e-5 * 90 = 0.0018 < 0.003 round-trip.
    rates = pd.Series([2e-5] * 180)
    res = walk_forward_funding_harvest(
        rates,
        symbol="TINY/USDT:USDT",
        notional_usdt=1000.0,
        fee_config=FeeConfig(),
        cfg=WalkForwardConfig(
            train_settlements=90,
            test_settlements=90,
            step_settlements=90,
            min_abs_train_mean=1e-5,
            require_train_edge=True,
        ),
    )
    assert res.n_traded_folds == 0
    assert res.folds[0].skip_reason == "no_edge_vs_fees"
