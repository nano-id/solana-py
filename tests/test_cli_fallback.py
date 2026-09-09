"""CLI --config order-independence and exchange fallback tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from ccxt.base.errors import ExchangeNotAvailable

from pairs_trading.data.cli import parse_args
from pairs_trading.data.config import ExchangeConfig, load_config
from pairs_trading.data.exchange import connect_with_fallback, is_fallback_worthy


def test_config_flag_before_and_after_subcommand(tmp_path: Path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "exchange:\n  id: okx\n  fallbacks: []\nohlcv:\n  timeframe: 1h\n",
        encoding="utf-8",
    )

    before = parse_args(["--config", str(cfg), "sync", "--symbols", "BTC/USDT"])
    after = parse_args(["sync", "--config", str(cfg), "--symbols", "BTC/USDT"])

    assert before.config == cfg
    assert after.config == cfg
    assert before.symbols == ["BTC/USDT"]
    assert after.symbols == ["BTC/USDT"]
    assert load_config(before.config).exchange.id == "okx"
    assert load_config(after.config).exchange.id == "okx"


def test_config_verbose_defaults_when_omitted():
    args = parse_args(["universe"])
    assert args.config is None
    assert args.verbose is False


def test_is_fallback_worthy_451():
    exc = ExchangeNotAvailable(
        "binance GET https://api.binance.com/api/v3/exchangeInfo 451 restricted location"
    )
    assert is_fallback_worthy(exc)


def test_connect_with_fallback_skips_blocked_primary():
    cfg = ExchangeConfig(id="binance", fallbacks=["okx"])

    bad = MagicMock()
    bad.load_markets.side_effect = ExchangeNotAvailable("451 restricted location")

    good = MagicMock()
    good.load_markets.return_value = {"BTC/USDT": {}}

    with patch(
        "pairs_trading.data.exchange.create_exchange",
        side_effect=[bad, good],
    ) as create:
        connected = connect_with_fallback(cfg)
        assert connected.exchange_id == "okx"
        assert connected.attempted == ("binance", "okx")
        assert create.call_count == 2
        good.load_markets.assert_called_once()


def test_candidate_ids_dedupes():
    cfg = ExchangeConfig(id="okx", fallbacks=["binance", "okx", "bybit"])
    assert cfg.candidate_ids() == ["okx", "binance", "bybit"]
