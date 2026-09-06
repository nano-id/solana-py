"""Live basis (perp vs index / spot) snapshots for cash-and-carry screen."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from pairs_trading.funding.symbols import swap_to_spot

logger = logging.getLogger(__name__)


def fetch_basis_snapshot(exchange: Any, swap_symbol: str) -> dict[str, float | str | None]:
    out: dict[str, float | str | None] = {
        "symbol": swap_symbol,
        "spot_symbol": swap_to_spot(swap_symbol),
        "funding_rate": None,
        "mark_price": None,
        "index_price": None,
        "spot_last": None,
        "quote_volume_24h": None,
        "basis_mark_index": None,
        "basis_mark_spot": None,
        "basis_bps_index": None,
        "basis_bps_spot": None,
    }
    try:
        fr = exchange.fetch_funding_rate(swap_symbol)
        out["funding_rate"] = _f(fr.get("fundingRate"))
        mark = fr.get("markPrice")
        idx = fr.get("indexPrice")
        info = fr.get("info") if isinstance(fr.get("info"), dict) else {}
        if mark is None:
            mark = info.get("markPx")
        if idx is None:
            idx = info.get("idxPx")
        out["mark_price"] = _f(mark)
        out["index_price"] = _f(idx)
    except Exception as exc:  # noqa: BLE001
        logger.warning("funding_rate failed for %s: %s", swap_symbol, exc)

    try:
        tick = exchange.fetch_ticker(swap_symbol)
        if out["mark_price"] is None:
            out["mark_price"] = _f(tick.get("last") or tick.get("close"))
        out["quote_volume_24h"] = _quote_volume_24h(tick)
    except Exception as exc:  # noqa: BLE001
        logger.warning("swap ticker failed for %s: %s", swap_symbol, exc)

    try:
        spot_tick = exchange.fetch_ticker(str(out["spot_symbol"]))
        out["spot_last"] = _f(spot_tick.get("last") or spot_tick.get("close"))
    except Exception:
        out["spot_last"] = None

    mark, idx, spot_px = out["mark_price"], out["index_price"], out["spot_last"]
    if mark is not None and idx is not None and float(idx) != 0:
        basis = float(mark) / float(idx) - 1.0
        out["basis_mark_index"] = basis
        out["basis_bps_index"] = basis * 10_000.0
    if mark is not None and spot_px is not None and float(spot_px) != 0:
        basis_s = float(mark) / float(spot_px) - 1.0
        out["basis_mark_spot"] = basis_s
        out["basis_bps_spot"] = basis_s * 10_000.0
    return out


def fetch_basis_table(exchange: Any, symbols: list[str]) -> pd.DataFrame:
    return pd.DataFrame([fetch_basis_snapshot(exchange, s) for s in symbols])


def _quote_volume_24h(tick: dict[str, Any]) -> float | None:
    """CCXT quoteVolume, else OKX volCcy24h, else baseVolume × last."""
    qv = _f(tick.get("quoteVolume"))
    if qv is not None:
        return qv
    info = tick.get("info") if isinstance(tick.get("info"), dict) else {}
    qv = _f(info.get("volCcy24h") or info.get("quoteVolume"))
    if qv is not None:
        return qv
    base = _f(tick.get("baseVolume") or info.get("vol24h"))
    last = _f(tick.get("last") or tick.get("close"))
    if base is not None and last is not None:
        return base * last
    return None


def _f(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
