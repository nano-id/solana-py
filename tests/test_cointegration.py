"""Cointegration engine unit tests + BTC-ETH smoke."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pairs_trading.cointegration.engine import engle_granger_pair, evaluate_pair
from pairs_trading.cointegration.half_life import estimate_half_life
from pairs_trading.cointegration.persist import save_scan_results


def _ohlcv_from_close(
    close: np.ndarray, start_ms: int = 1_700_000_000_000, step_ms: int = 86_400_000
) -> pd.DataFrame:
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


def test_synthetic_cointegrated_pair_log_space():
    """Generate cointegration in log space; default path must recover β ≈ 1.1."""
    rng = np.random.default_rng(42)
    n = 400
    log_x = np.cumsum(rng.normal(0, 0.01, size=n)) + 4.0
    log_y = 1.1 * log_x + rng.normal(0, 0.005, size=n)
    x = np.exp(log_x)
    y = np.exp(log_y)
    hit = engle_granger_pair(
        "Y/USDT",
        "X/USDT",
        _ohlcv_from_close(y),
        _ohlcv_from_close(x),
        pvalue_threshold=0.05,
        min_obs=100,
        price_space="log",
    )
    assert hit is not None
    assert hit.price_space == "log"
    assert hit.pvalue < 0.05
    assert hit.hedge_ratio == pytest.approx(1.1, rel=0.05)
    assert hit.half_life < float("inf")


def test_level_beta_collapses_on_scale_difference():
    """Sanity: raw-price OLS on y≈0.05·x yields β≈0.05 (scale artifact)."""
    rng = np.random.default_rng(0)
    n = 300
    x = 50_000 + np.cumsum(rng.normal(0, 200, size=n))
    y = 0.05 * x + rng.normal(0, 20, size=n)
    level = evaluate_pair(
        "ETH/USDT",
        "BTC/USDT",
        _ohlcv_from_close(y),
        _ohlcv_from_close(x),
        min_obs=100,
        price_space="level",
    )
    log = evaluate_pair(
        "ETH/USDT",
        "BTC/USDT",
        _ohlcv_from_close(np.clip(y, 1e-6, None)),
        _ohlcv_from_close(np.clip(x, 1e-6, None)),
        min_obs=100,
        price_space="log",
    )
    assert level.price_space == "level"
    assert log.price_space == "log"
    assert level.hedge_ratio == pytest.approx(0.05, rel=0.1)
    # Log β is an elasticity, not forced to equal the level scale factor.
    assert abs(log.hedge_ratio - level.hedge_ratio) > 0.2


def test_save_scan_results_dated_and_latest(tmp_path):
    df = pd.DataFrame(
        [
            {
                "y_symbol": "ETH/USDT",
                "x_symbol": "BTC/USDT",
                "pvalue": 0.01,
                "test_statistic": -4.0,
                "critical_value_5pct": -3.3,
                "hedge_ratio": 0.9,
                "intercept": 0.1,
                "half_life": 24.0,
                "n_obs": 200,
                "price_space": "log",
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
def test_btc_eth_log_space_smoke():
    """Live smoke on log prices — report p / β / half-life; do not require p<0.05.

    Cointegration is regime- and window-dependent. With ~1y daily data, BTC-ETH
    often fails EG on log prices even when level-price OLS looks 'significant'
    (scale artifact). This test locks the pipeline to log space and asserts the
    elasticity β is in a sane band; p-value is reported for human review.
    """
    from pairs_trading.data.config import load_config
    from pairs_trading.data.pipeline import DataPipeline
    from pairs_trading.data.storage import ParquetStore

    config = load_config("config/smoke.yaml")
    assert config.cointegration.price_space == "log"
    pipe = DataPipeline(config)
    pipe.sync_symbol("BTC/USDT", force_full=True)
    pipe.sync_symbol("ETH/USDT", force_full=True)

    store = ParquetStore(config.storage.root, pipe.exchange_id, config.ohlcv.timeframe)
    btc = store.load("BTC/USDT")
    eth = store.load("ETH/USDT")
    assert len(btc) >= config.cointegration.min_obs
    assert len(eth) >= config.cointegration.min_obs

    log_hit = evaluate_pair(
        "ETH/USDT",
        "BTC/USDT",
        eth,
        btc,
        min_obs=config.cointegration.min_obs,
        price_space="log",
    )
    level_hit = evaluate_pair(
        "ETH/USDT",
        "BTC/USDT",
        eth,
        btc,
        min_obs=config.cointegration.min_obs,
        price_space="level",
    )
    print(
        f"\nBTC-ETH via {pipe.exchange_id} n={log_hit.n_obs}\n"
        f"  [log]   p={log_hit.pvalue:.6f} β={log_hit.hedge_ratio:.6f} "
        f"half_life={log_hit.half_life:.2f}\n"
        f"  [level] p={level_hit.pvalue:.6f} β={level_hit.hedge_ratio:.6f} "
        f"half_life={level_hit.half_life:.2f}"
    )
    assert log_hit.price_space == "log"
    # Elasticity should be O(1), not the ~0.05 scale artifact of raw prices.
    assert 0.3 < log_hit.hedge_ratio < 2.5
    assert abs(log_hit.hedge_ratio - level_hit.hedge_ratio) > 0.2
