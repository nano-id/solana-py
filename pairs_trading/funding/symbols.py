"""Symbol helpers for swap ↔ spot funding harvest legs."""

from __future__ import annotations


def swap_to_spot(symbol: str) -> str:
    """BTC/USDT:USDT → BTC/USDT; leave spot symbols unchanged."""
    if ":" in symbol:
        return symbol.split(":", 1)[0]
    return symbol


def spot_to_swap(symbol: str, settle: str | None = None) -> str:
    """BTC/USDT → BTC/USDT:USDT (linear USDT-m default)."""
    if ":" in symbol:
        return symbol
    quote = symbol.split("/")[-1] if "/" in symbol else "USDT"
    settle = settle or quote
    return f"{symbol}:{settle}"


def base_quote(symbol: str) -> tuple[str, str]:
    spot = swap_to_spot(symbol)
    if "/" not in spot:
        raise ValueError(f"Unrecognized symbol: {symbol}")
    base, quote = spot.split("/", 1)
    return base, quote
