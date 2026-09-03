"""Cointegration engine unit tests + BTC-ETH smoke."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pairs_trading.cointegration.engine import engle_granger_pair, evaluate_pair
from pairs_trading.cointegration.half_life import estimate_half_life
from pairs_trading.cointegration.persist import save_scan_results


def _ohlcv_from_close(close: np.ndarray, start_ms: int = 1_700_000_000_000, step_ms: int = 86_400_000) -> pd.DataFrame:
    n = len(close)
    ts = np.arange(n) * step_ms + start_ms
    return pd.DataFrame(
        {
            "timestamp": ts.astype(np.int64),
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.ones(n),
        }
    )


def test_half_life_known_ar1():
    # φ = 2^(-1/10) => half-life ≈ 10
    phi = 2 ** (-1 / 10)
    rng = np.random.default_rng(0)
    n = 2000
    eps = rng.normal(0, 0.01, size=n)
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = phi * s[t - 1] + eps[t]
    hl = estimate_half_life(s)
    assert 7 < hl < 14


def test_synthetic_cointegrated_pair_detected():
    rng = np.random.default_rng(42)
    n = 400
    x = np.cumsum(rng.normal(0, 1, size=n)) + 100
    # y = 2 x + stationary noise  → strongly cointegrated
    y = 2.0 * x + rng.normal(0, 0.5, size=n)
    hit = engle_granger_pair(
        "Y/USDT",
        "X/USDT",
        _ohlcv_from_close(y),
        _ohlcv_from_close(x),
        pvalue_threshold=0.05,
        min_obs=100,
    )
    assert hit is not None
    assert hit.pvalue < 0.05
    assert hit.hedge_ratio == pytest.approx(2.0, rel=0.05)
    assert hit.half_life < float("inf")


def test_save_scan_results_dated_and_latest(tmp_path):
    df = pd.DataFrame(
        [
            {
                "y_symbol": "ETH/USDT",
                "x_symbol": "BTC/USDT",
                "pvalue": 0.01,
                "test_statistic": -4.0,
                "critical_value_5pct": -3.3,
                "hedge_ratio": 0.05,
                "intercept": 1.0,
                "half_life": 24.0,
                "n_obs": 200,
            }
        ]
    )
    paths = save_scan_results(
        df,
        root=tmp_path,
        exchange_id="okx",
        timeframe="1d",
        mode="both",
        meta={"tested_pairs": 1},
    )
    assert paths["dated"].is_file()
    assert paths["latest"].is_file()
    assert paths["latest_meta"].is_file()


@pytest.mark.integration
def test_btc_eth_cointegration_smoke():
    """Live smoke: BTC-ETH often cointegrates on daily data; assert p < 0.05.

    Requires network. Uses config-driven exchange with geo-fallback.
    """
    from pairs_trading.data.config import load_config
    from pairs_trading.data.pipeline import DataPipeline
    from pairs_trading.data.storage import ParquetStore

    config = load_config("config/smoke.yaml")
    pipe = DataPipeline(config)
    pipe.sync_symbol("BTC/USDT", force_full=True)
    pipe.sync_symbol("ETH/USDT", force_full=True)

    store = ParquetStore(config.storage.root, pipe.exchange_id, config.ohlcv.timeframe)
    btc = store.load("BTC/USDT")
    eth = store.load("ETH/USDT")
    assert len(btc) >= config.cointegration.min_obs
    assert len(eth) >= config.cointegration.min_obs

    # Literature convention often treats ETH ~ β BTC
    hit = evaluate_pair(
        "ETH/USDT",
        "BTC/USDT",
        eth,
        btc,
        min_obs=config.cointegration.min_obs,
    )
    print(
        f"\nBTC-ETH smoke via {pipe.exchange_id}: "
        f"p={hit.pvalue:.6f} β={hit.hedge_ratio:.6f} "
        f"half_life={hit.half_life:.2f} bars n={hit.n_obs}"
    )
    assert hit.pvalue < 0.05, (
        f"BTC-ETH Engle-Granger p={hit.pvalue:.4f} not < 0.05 on this window "
        f"(exchange={pipe.exchange_id}, n={hit.n_obs}). Review regime/window."
    )
