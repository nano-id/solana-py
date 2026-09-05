"""Point-in-time aware trading universe selection.

Survivorship bias control: universe snapshots are dated. Historical backtests
must load the snapshot that would have been known at that as-of time — never
today's top coins projected backwards.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ccxt

from pairs_trading.data.config import ExchangeConfig, UniverseConfig


@dataclass(frozen=True)
class UniverseMember:
    symbol: str
    base: str
    quote: str
    quote_volume_24h: float
    last_price: float | None
    rank: int


@dataclass(frozen=True)
class UniverseSnapshot:
    exchange_id: str
    market_type: str
    quote: str
    as_of: str  # ISO-8601 UTC
    top_n: int
    members: list[UniverseMember]

    @property
    def symbols(self) -> list[str]:
        return [m.symbol for m in self.members]


def _is_active_target_market(
    market: dict[str, Any],
    *,
    quote: str,
    market_type: str,
    exclude_bases: set[str],
) -> bool:
    if not market.get("active", True):
        return False
    if market.get("quote") != quote:
        return False
    base = str(market.get("base") or "").upper()
    if base in exclude_bases:
        return False

    # Spot vs swap/perpetual filter via CCXT market flags.
    if market_type == "spot":
        if not market.get("spot", False):
            return False
    elif market_type == "swap":
        if not (market.get("swap") or market.get("future")):
            return False
        # Prefer linear USDT-settled perps when the flag exists.
        if market.get("linear") is False:
            return False
    return True


def select_universe(
    exchange: ccxt.Exchange,
    exchange_config: ExchangeConfig,
    universe_config: UniverseConfig,
    *,
    as_of: datetime | None = None,
) -> UniverseSnapshot:
    """Select top-N symbols by 24h quote volume from current tickers.

    Note: exchange tickers reflect *current* rankings. For true historical
    point-in-time universes, persist snapshots regularly and reload by date.
    """
    if as_of is None:
        as_of = datetime.now(timezone.utc)
    elif as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)

    markets = exchange.load_markets()
    tickers = exchange.fetch_tickers()
    exclude = set(universe_config.exclude_bases)

    candidates: list[tuple[float, str, dict[str, Any], dict[str, Any]]] = []
    for symbol, market in markets.items():
        if not _is_active_target_market(
            market,
            quote=exchange_config.quote,
            market_type=exchange_config.market_type,
            exclude_bases=exclude,
        ):
            continue
        ticker = tickers.get(symbol) or {}
        quote_volume = ticker.get("quoteVolume")
        if quote_volume is None:
            # Fallback: baseVolume * last
            base_volume = ticker.get("baseVolume")
            last = ticker.get("last")
            if base_volume is not None and last is not None:
                quote_volume = float(base_volume) * float(last)
            else:
                continue
        quote_volume = float(quote_volume)
        if quote_volume < universe_config.min_quote_volume:
            continue
        candidates.append((quote_volume, symbol, market, ticker))

    candidates.sort(key=lambda row: row[0], reverse=True)
    top = candidates[: universe_config.top_n]

    members = [
        UniverseMember(
            symbol=symbol,
            base=str(market.get("base") or ""),
            quote=str(market.get("quote") or exchange_config.quote),
            quote_volume_24h=quote_volume,
            last_price=(float(ticker["last"]) if ticker.get("last") is not None else None),
            rank=idx + 1,
        )
        for idx, (quote_volume, symbol, market, ticker) in enumerate(top)
    ]

    return UniverseSnapshot(
        exchange_id=exchange_config.id,
        market_type=exchange_config.market_type,
        quote=exchange_config.quote,
        as_of=as_of.astimezone(timezone.utc).isoformat(),
        top_n=universe_config.top_n,
        members=members,
    )


def save_universe_snapshot(snapshot: UniverseSnapshot, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "exchange_id": snapshot.exchange_id,
        "market_type": snapshot.market_type,
        "quote": snapshot.quote,
        "as_of": snapshot.as_of,
        "top_n": snapshot.top_n,
        "members": [asdict(m) for m in snapshot.members],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_universe_snapshot(path: Path) -> UniverseSnapshot:
    payload = json.loads(path.read_text(encoding="utf-8"))
    members = [UniverseMember(**m) for m in payload["members"]]
    return UniverseSnapshot(
        exchange_id=payload["exchange_id"],
        market_type=payload["market_type"],
        quote=payload["quote"],
        as_of=payload["as_of"],
        top_n=int(payload["top_n"]),
        members=members,
    )


def universe_snapshot_path(root: Path, exchange_id: str, as_of: datetime | None = None) -> Path:
    """Dated snapshot path: data/{exchange}/universe/{YYYY-MM-DD}.json"""
    if as_of is None:
        as_of = datetime.now(timezone.utc)
    day = as_of.astimezone(timezone.utc).strftime("%Y-%m-%d")
    return root / exchange_id / "universe" / f"{day}.json"
