"""End-to-end data pipeline: universe snapshot + OHLCV sync."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pairs_trading.data.config import DataConfig, load_config
from pairs_trading.data.exchange import create_exchange
from pairs_trading.data.fetcher import fetch_ohlcv_range, lookback_start_ms, timeframe_to_milliseconds
from pairs_trading.data.storage import ParquetStore
from pairs_trading.data.universe import (
    UniverseSnapshot,
    save_universe_snapshot,
    select_universe,
    universe_snapshot_path,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SymbolSyncResult:
    symbol: str
    bars_written: int
    path: str
    since_ms: int
    until_ms: int


@dataclass(frozen=True)
class PipelineResult:
    universe: UniverseSnapshot | None
    universe_path: str | None
    synced: list[SymbolSyncResult]


class DataPipeline:
    """Orchestrates exchange → universe snapshot → parquet OHLCV updates."""

    def __init__(self, config: DataConfig | None = None) -> None:
        self.config = config or load_config()
        self.exchange = create_exchange(self.config.exchange)
        self.store = ParquetStore(
            root=self.config.storage.root,
            exchange_id=self.config.exchange.id,
            timeframe=self.config.ohlcv.timeframe,
        )

    def snapshot_universe(self, *, as_of: datetime | None = None) -> tuple[UniverseSnapshot, Path]:
        as_of = as_of or datetime.now(timezone.utc)
        snapshot = select_universe(
            self.exchange,
            self.config.exchange,
            self.config.universe,
            as_of=as_of,
        )
        path = universe_snapshot_path(self.config.storage.root, self.config.exchange.id, as_of)
        save_universe_snapshot(snapshot, path)
        logger.info(
            "Universe snapshot %s: %d symbols -> %s",
            snapshot.as_of,
            len(snapshot.members),
            path,
        )
        return snapshot, path

    def sync_symbol(
        self,
        symbol: str,
        *,
        until_ms: int | None = None,
        force_full: bool = False,
    ) -> SymbolSyncResult:
        until_ms = until_ms or int(datetime.now(timezone.utc).timestamp() * 1000)
        timeframe_ms = timeframe_to_milliseconds(self.exchange, self.config.ohlcv.timeframe)

        if force_full or not self.store.exists(symbol):
            since_ms = lookback_start_ms(self.config.ohlcv.lookback_days)
        else:
            latest = self.store.latest_timestamp_ms(symbol)
            since_ms = (
                latest
                if latest is not None
                else lookback_start_ms(self.config.ohlcv.lookback_days)
            )
            # Overlap one candle to heal gap/late-revision edge cases.
            since_ms = max(0, since_ms - timeframe_ms)

        df = fetch_ohlcv_range(
            self.exchange,
            symbol,
            self.config.ohlcv,
            since_ms=since_ms,
            until_ms=until_ms,
            closed_only=True,
        )
        merged = self.store.upsert(symbol, df)
        path = self.store.path_for(symbol)
        logger.info(
            "%s: upserted %d new rows, total %d -> %s",
            symbol,
            len(df),
            len(merged),
            path,
        )
        return SymbolSyncResult(
            symbol=symbol,
            bars_written=len(df),
            path=str(path),
            since_ms=since_ms,
            until_ms=until_ms,
        )

    def run(
        self,
        *,
        symbols: list[str] | None = None,
        refresh_universe: bool = True,
        force_full: bool = False,
    ) -> PipelineResult:
        """Optionally snapshot universe, then sync OHLCV for each symbol."""
        snapshot: UniverseSnapshot | None = None
        universe_path: Path | None = None

        if symbols is not None:
            sync_list = symbols
            if refresh_universe:
                snapshot, universe_path = self.snapshot_universe()
        else:
            snapshot, universe_path = self.snapshot_universe()
            sync_list = snapshot.symbols

        synced: list[SymbolSyncResult] = []
        for symbol in sync_list:
            try:
                synced.append(self.sync_symbol(symbol, force_full=force_full))
            except Exception:
                logger.exception("Failed syncing %s — continuing", symbol)

        return PipelineResult(
            universe=snapshot,
            universe_path=str(universe_path) if universe_path else None,
            synced=synced,
        )
