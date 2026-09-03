"""Pairwise cointegration scan over a point-in-time universe."""

from __future__ import annotations

import itertools
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pairs_trading.cointegration.engine import CointegrationHit, engle_granger_pair
from pairs_trading.cointegration.persist import save_scan_results
from pairs_trading.data.config import DataConfig, load_config
from pairs_trading.data.storage import ParquetStore
from pairs_trading.data.universe import load_universe_snapshot

logger = logging.getLogger(__name__)

__all__ = ["CointegrationHit", "resolve_universe_symbols", "scan_pairs", "engle_granger_pair"]


def _parse_day(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def list_universe_snapshot_paths(root: Path, exchange_id: str) -> list[Path]:
    folder = root / exchange_id / "universe"
    if not folder.is_dir():
        return []
    return sorted(folder.glob("*.json"))


def resolve_universe_symbols(
    config: DataConfig,
    store: ParquetStore,
    *,
    symbols: list[str] | None = None,
    universe_as_of: str | None = None,
) -> tuple[list[str], str]:
    """Pick symbols without survivorship leakage.

    - If `symbols` given: use them (caller responsibility).
    - Else load the universe snapshot on/before `universe_as_of`.
    - `universe_as_of="auto"`: use the earliest snapshot on disk whose date is
      <= the earliest common bar date across stored series; if no snapshots,
      fall back to symbols present in the parquet store (logged as weak PIT).
    """
    as_of = universe_as_of or config.cointegration.universe_as_of

    if symbols is not None:
        return symbols, f"explicit:{len(symbols)}"

    snapshots = list_universe_snapshot_paths(config.storage.root, store.exchange_id)
    if not snapshots:
        stored = store.list_symbols()
        # Filenames are sanitized; cannot reliably reverse to CCXT symbols.
        logger.warning(
            "No universe snapshots found — cannot enforce point-in-time universe. "
            "Pass --symbols or run pairs-data universe first. Stored files: %s",
            stored,
        )
        raise FileNotFoundError(
            f"No universe snapshots under {config.storage.root / store.exchange_id / 'universe'}"
        )

    if as_of == "auto":
        # Prefer the oldest snapshot (closest to sample start) to avoid using
        # today's ranking for a historical window.
        path = snapshots[0]
        reason = f"auto:oldest_snapshot:{path.stem}"
    else:
        target = _parse_day(as_of)
        eligible = [p for p in snapshots if _parse_day(p.stem) <= target]
        if not eligible:
            raise FileNotFoundError(
                f"No universe snapshot on or before {as_of} "
                f"(have {[p.stem for p in snapshots]})"
            )
        path = eligible[-1]
        reason = f"as_of:{as_of}:using:{path.stem}"

    snap = load_universe_snapshot(path)
    return snap.symbols, reason


def load_symbol_frame(store: ParquetStore, symbol: str) -> pd.DataFrame:
    df = store.load(symbol)
    if df.empty:
        raise FileNotFoundError(f"No OHLCV for {symbol} in {store.base_dir}")
    return df


def scan_pairs(
    config: DataConfig | None = None,
    *,
    symbols: list[str] | None = None,
    exchange_id: str | None = None,
    universe_as_of: str | None = None,
    persist: bool = True,
) -> tuple[pd.DataFrame, list[CointegrationHit]]:
    """Test all unordered pairs in the PIT universe; return hits DataFrame."""
    config = config or load_config()
    eid = exchange_id or config.exchange.id
    store = ParquetStore(config.storage.root, eid, config.ohlcv.timeframe)

    universe, universe_reason = resolve_universe_symbols(
        config, store, symbols=symbols, universe_as_of=universe_as_of
    )
    logger.info(
        "Cointegration scan on %s (%s) — %d symbols via %s",
        eid,
        config.ohlcv.timeframe,
        len(universe),
        universe_reason,
    )

    # Preload
    frames: dict[str, pd.DataFrame] = {}
    for sym in universe:
        try:
            frames[sym] = load_symbol_frame(store, sym)
        except FileNotFoundError:
            logger.warning("Skipping %s — no local OHLCV", sym)

    hits: list[CointegrationHit] = []
    tested = 0
    for y_sym, x_sym in itertools.combinations(sorted(frames.keys()), 2):
        tested += 1
        try:
            hit = engle_granger_pair(
                y_sym,
                x_sym,
                frames[y_sym],
                frames[x_sym],
                pvalue_threshold=config.cointegration.pvalue_threshold,
                min_obs=config.cointegration.min_obs,
            )
        except ValueError as exc:
            logger.debug("Skip %s-%s: %s", y_sym, x_sym, exc)
            continue
        if hit is not None:
            hits.append(hit)

    rows = [asdict(h) for h in hits]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("pvalue").reset_index(drop=True)

    logger.info(
        "Scan complete: tested=%d hits=%d (α=%.3f)",
        tested,
        len(hits),
        config.cointegration.pvalue_threshold,
    )

    if persist:
        save_scan_results(
            df,
            root=config.storage.root,
            exchange_id=eid,
            timeframe=config.ohlcv.timeframe,
            mode=config.cointegration.persist_mode,
            meta={
                "universe_reason": universe_reason,
                "symbols": list(frames.keys()),
                "tested_pairs": tested,
                "pvalue_threshold": config.cointegration.pvalue_threshold,
                "min_obs": config.cointegration.min_obs,
            },
        )

    return df, hits
