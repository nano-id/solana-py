"""Funding-rate sync pipeline (Module 1 data layer)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from pairs_trading.data.exchange import connect_with_fallback
from pairs_trading.funding.config import FundingAppConfig, load_funding_config
from pairs_trading.funding.fetcher import default_since_ms, fetch_funding_rate_history
from pairs_trading.funding.storage import FundingStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FundingSyncResult:
    symbol: str
    rows_written: int
    path: str
    exchange_id: str


class FundingPipeline:
    """Connect (swap) → paginate funding history → parquet upsert."""

    def __init__(self, config: FundingAppConfig | None = None) -> None:
        self.config = config or load_funding_config()
        connected = connect_with_fallback(self.config.exchange)
        self.exchange = connected.exchange
        self.exchange_id = connected.exchange_id
        self.config.exchange.id = self.exchange_id
        self.store = FundingStore(self.config.storage.root, self.exchange_id)
        logger.info(
            "FundingPipeline ready — '%s' (tried %s)",
            self.exchange_id,
            " → ".join(connected.attempted),
        )

    def resolve_symbols(self, symbols: list[str] | None = None) -> list[str]:
        if symbols:
            return list(symbols)
        return list(self.config.funding.default_symbols)

    def sync_symbol(self, symbol: str, *, force_full: bool = False) -> FundingSyncResult:
        until_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        if force_full or not self.store.exists(symbol):
            since_ms = default_since_ms(self.config.funding.lookback_days, None)
        else:
            since_ms = default_since_ms(
                self.config.funding.lookback_days,
                self.store.latest_timestamp_ms(symbol),
            )
        frame = fetch_funding_rate_history(
            self.exchange,
            symbol,
            since_ms=since_ms,
            until_ms=until_ms,
            limit_per_request=self.config.funding.limit_per_request,
        )
        if frame.empty:
            path = str(self.store.path_for(symbol))
            logger.warning("No new funding rows for %s", symbol)
            return FundingSyncResult(symbol, 0, path, self.exchange_id)
        merged = self.store.upsert(symbol, frame)
        path = str(self.store.path_for(symbol))
        logger.info(
            "[%s] %s: upserted %d rows, total %d → %s",
            self.exchange_id,
            symbol,
            len(frame),
            len(merged),
            path,
        )
        return FundingSyncResult(symbol, len(frame), path, self.exchange_id)

    def sync_many(
        self,
        symbols: list[str] | None = None,
        *,
        force_full: bool = False,
    ) -> list[FundingSyncResult]:
        out: list[FundingSyncResult] = []
        for sym in self.resolve_symbols(symbols):
            try:
                out.append(self.sync_symbol(sym, force_full=force_full))
            except Exception:
                logger.exception("Funding sync failed for %s", sym)
        return out
