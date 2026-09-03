"""Smoke tests for exchange factory (no network I/O)."""

from __future__ import annotations

import pytest

from pairs_trading.data.config import ExchangeConfig
from pairs_trading.data.exchange import create_exchange


def test_create_binance_spot():
    ex = create_exchange(ExchangeConfig(id="binance", market_type="spot"))
    assert ex.id == "binance"
    assert ex.options.get("defaultType") == "spot"


def test_create_bybit_swap():
    ex = create_exchange(ExchangeConfig(id="bybit", market_type="swap"))
    assert ex.id == "bybit"
    assert ex.options.get("defaultType") == "swap"


def test_create_okx_spot():
    ex = create_exchange(ExchangeConfig(id="okx", market_type="spot"))
    assert ex.id == "okx"
    assert ex.options.get("defaultType") == "spot"


def test_create_exchange_id_override():
    cfg = ExchangeConfig(id="binance", market_type="spot")
    ex = create_exchange(cfg, exchange_id="okx")
    assert ex.id == "okx"


def test_unsupported_exchange():
    with pytest.raises(ValueError, match="Unsupported"):
        create_exchange(ExchangeConfig(id="kraken"))
