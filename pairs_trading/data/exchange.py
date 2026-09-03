"""CCXT exchange factory and thin market-data helpers."""

from __future__ import annotations

from typing import Any

import ccxt

from pairs_trading.data.config import ExchangeConfig


# Primary targets: binance, bybit. okx included as a geo-fallback when those
# block the runtime IP (HTTP 451 / CloudFront 403).
SUPPORTED_EXCHANGES = ("binance", "bybit", "okx")


def create_exchange(config: ExchangeConfig, api_key: str | None = None, secret: str | None = None) -> ccxt.Exchange:
    """Build a rate-limited public (or optionally authenticated) CCXT client."""
    exchange_id = config.id.lower()
    if exchange_id not in SUPPORTED_EXCHANGES:
        raise ValueError(
            f"Unsupported exchange '{config.id}'. Supported: {', '.join(SUPPORTED_EXCHANGES)}"
        )
    if not hasattr(ccxt, exchange_id):
        raise ValueError(f"CCXT has no exchange class for '{exchange_id}'")

    klass = getattr(ccxt, exchange_id)
    params: dict[str, Any] = {
        "enableRateLimit": config.enable_rate_limit,
        "options": {},
    }
    if api_key:
        params["apiKey"] = api_key
    if secret:
        params["secret"] = secret

    if exchange_id == "binance":
        # Explicit defaultType so spot vs swap is unambiguous.
        params["options"]["defaultType"] = "future" if config.market_type == "swap" else "spot"
    elif exchange_id == "bybit":
        params["options"]["defaultType"] = "swap" if config.market_type == "swap" else "spot"
    elif exchange_id == "okx":
        # OKX uses defaultType spot|swap|future
        params["options"]["defaultType"] = "swap" if config.market_type == "swap" else "spot"

    exchange: ccxt.Exchange = klass(params)
    if config.sandbox and hasattr(exchange, "set_sandbox_mode"):
        exchange.set_sandbox_mode(True)
    return exchange


def load_markets(exchange: ccxt.Exchange) -> dict[str, Any]:
    """Fetch and cache markets on the exchange instance."""
    return exchange.load_markets()
