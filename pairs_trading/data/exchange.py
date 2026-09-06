"""CCXT exchange factory with config-driven selection and geo-fallback retry."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import ccxt
from ccxt.base.errors import (
    ExchangeError,
    ExchangeNotAvailable,
    NetworkError,
    RequestTimeout,
)

from pairs_trading.data.config import ExchangeConfig

logger = logging.getLogger(__name__)

# Registry of adapters we explicitly support. Selection still comes from config.
SUPPORTED_EXCHANGES = ("binance", "bybit", "okx")

# HTTP / CCXT signals that usually mean "try another venue", not a bad symbol.
_FALLBACK_ERROR_TYPES = (
    ExchangeNotAvailable,
    NetworkError,
    RequestTimeout,
    ExchangeError,
)
_FALLBACK_STATUS_MARKERS = ("451", "403", "restricted location", "cloudfront", "forbidden")


@dataclass(frozen=True)
class ConnectedExchange:
    exchange: ccxt.Exchange
    exchange_id: str
    attempted: tuple[str, ...]


def create_exchange(
    config: ExchangeConfig,
    *,
    exchange_id: str | None = None,
    api_key: str | None = None,
    secret: str | None = None,
) -> ccxt.Exchange:
    """Build a rate-limited CCXT client for the given (or config) exchange id."""
    eid = (exchange_id or config.id).lower()
    if eid not in SUPPORTED_EXCHANGES:
        raise ValueError(
            f"Unsupported exchange '{eid}'. Supported: {', '.join(SUPPORTED_EXCHANGES)}"
        )
    if not hasattr(ccxt, eid):
        raise ValueError(f"CCXT has no exchange class for '{eid}'")

    klass = getattr(ccxt, eid)
    params: dict[str, Any] = {
        "enableRateLimit": config.enable_rate_limit,
        "options": {},
    }
    if api_key:
        params["apiKey"] = api_key
    if secret:
        params["secret"] = secret

    # defaultType is exchange-specific but always driven by config.market_type.
    params["options"]["defaultType"] = "swap" if config.market_type == "swap" else "spot"

    exchange: ccxt.Exchange = klass(params)
    if config.sandbox and hasattr(exchange, "set_sandbox_mode"):
        exchange.set_sandbox_mode(True)
    return exchange


def is_fallback_worthy(exc: BaseException) -> bool:
    """True when the failure looks like venue/geo unavailability."""
    if isinstance(exc, _FALLBACK_ERROR_TYPES):
        msg = str(exc).lower()
        # Soft: any ExchangeNotAvailable / NetworkError is fallback-worthy.
        if isinstance(exc, (ExchangeNotAvailable, NetworkError, RequestTimeout)):
            return True
        return any(marker in msg for marker in _FALLBACK_STATUS_MARKERS)
    msg = str(exc).lower()
    return any(marker in msg for marker in _FALLBACK_STATUS_MARKERS)


def connect_with_fallback(
    config: ExchangeConfig,
    *,
    api_key: str | None = None,
    secret: str | None = None,
    probe: bool = True,
) -> ConnectedExchange:
    """Try primary then fallbacks until load_markets (or construct) succeeds.

    Logs which venue ultimately served the data so audits stay honest.
    """
    attempted: list[str] = []
    errors: list[str] = []

    for eid in config.candidate_ids():
        attempted.append(eid)
        try:
            exchange = create_exchange(
                config, exchange_id=eid, api_key=api_key, secret=secret
            )
            if probe:
                exchange.load_markets()
            logger.info("Connected to exchange '%s' (data source)", eid)
            return ConnectedExchange(
                exchange=exchange,
                exchange_id=eid,
                attempted=tuple(attempted),
            )
        except Exception as exc:  # noqa: BLE001 — decide fallback vs raise
            errors.append(f"{eid}: {exc}")
            if is_fallback_worthy(exc):
                logger.warning(
                    "Exchange '%s' unavailable (%s) — trying next fallback",
                    eid,
                    exc,
                )
                continue
            raise

    raise RuntimeError(
        "All configured exchanges failed. Attempted: "
        + "; ".join(errors)
    )


def load_markets(exchange: ccxt.Exchange) -> dict[str, Any]:
    """Fetch and cache markets on the exchange instance."""
    return exchange.load_markets()
