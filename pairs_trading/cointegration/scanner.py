"""Pairwise cointegration scan over a point-in-time universe."""

from __future__ import annotations

import itertools
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pairs_trading.cointegration.engine import CointegrationHit, evaluate_pair
from pairs_trading.cointegration.persist import save_scan_results
from pairs_trading.data.config import DataConfig, load_config
from pairs_trading.data.storage import ParquetStore
from pairs_trading.data.universe import load_universe_snapshot

logger = logging.getLogger(__name__)

__all__ = ["CointegrationHit", "resolve_universe_symbols", "scan_pairs"]


def _parse_day(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def list_universe_snapshot_paths(root: Path, exchange_id: str) -> list[Path]:
    folder = root / exchange_id / "universe"
    if not folder.is_dir():
        return []
    return sorted(folder.glob("*.json"))


def resolve_exchange_id(config: DataConfig, exchange_id: str | None = None) -> str:
    """Prefer explicit id; else first candidate that has a universe snapshot or OHLCV."""
    if exchange_id:
        return exchange_id
    root = Path(config.storage.root)
    for eid in config.exchange.candidate_ids():
        uni = root / eid / "universe"
        ohlcv = root / eid / config.ohlcv.timeframe
        if uni.is_dir() and any(uni.glob("*.json")):
            return eid
        if ohlcv.is_dir() and any(ohlcv.glob("*.parquet")):
            return eid
    return config.exchange.id


def resolve_universe_symbols(
    config: DataConfig,
    store: ParquetStore,
    *,
    symbols: list[str] | None = None,
    universe_as_of: str | None = None,
) -> tuple[list[str], str]:
    """Pick symbols without survivorship leakage when snapshots exist."""
    as_of = universe_as_of or config.cointegration.universe_as_of

    if symbols is not None:
        return symbols, f"explicit:{len(symbols)}"

    snapshots = list_universe_snapshot_paths(config.storage.root, store.exchange_id)
    if not snapshots:
        raise FileNotFoundError(
            f"No universe snapshots under "
            f"{config.storage.root / store.exchange_id / 'universe'}. "
            "Run: pairs-data universe  (or pairs-data sync)"
        )

    if as_of == "auto":
        # Newest snapshot = current liquid set for a live scan.
        # (True historical PIT needs dated snapshots taken over time.)
        path = snapshots[-1]
        reason = f"auto:latest_snapshot:{path.stem}"
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
    return list(snap.symbols), reason


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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test all unordered pairs; return (full_ranked_df, significant_df).

    Full table is sorted by p-value ascending and includes `significant`
    (p < α). Does not drop non-hits — universe screening needs the full ranking.
    """
    config = config or load_config()
    eid = resolve_exchange_id(config, exchange_id)
    store = ParquetStore(config.storage.root, eid, config.ohlcv.timeframe)
    alpha = config.cointegration.pvalue_threshold

    universe, universe_reason = resolve_universe_symbols(
        config, store, symbols=symbols, universe_as_of=universe_as_of
    )
    logger.info(
        "Cointegration scan on %s (%s, price_space=%s) — %d symbols via %s",
        eid,
        config.ohlcv.timeframe,
        config.cointegration.price_space,
        len(universe),
        universe_reason,
    )

    frames: dict[str, pd.DataFrame] = {}
    for sym in universe:
        try:
            frames[sym] = load_symbol_frame(store, sym)
        except FileNotFoundError:
            logger.warning("Skipping %s — no local OHLCV", sym)

    rows: list[dict] = []
    tested = 0
    skipped = 0
    for y_sym, x_sym in itertools.combinations(sorted(frames.keys()), 2):
        tested += 1
        try:
            hit = evaluate_pair(
                y_sym,
                x_sym,
                frames[y_sym],
                frames[x_sym],
                min_obs=config.cointegration.min_obs,
                price_space=config.cointegration.price_space,
            )
        except ValueError as exc:
            skipped += 1
            logger.debug("Skip %s-%s: %s", y_sym, x_sym, exc)
            continue
        row = asdict(hit)
        row["significant"] = bool(hit.pvalue < alpha)
        row["pair"] = f"{hit.y_symbol}|{hit.x_symbol}"
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("pvalue", kind="mergesort").reset_index(drop=True)
        # Stable rank after sort (1 = strongest rejection of no-cointegration).
        df.insert(0, "rank", range(1, len(df) + 1))

    sig = df[df["significant"]].copy() if not df.empty else df.copy()

    logger.info(
        "Scan complete: tested=%d skipped=%d significant=%d (α=%.3f, price_space=%s)",
        tested,
        skipped,
        len(sig),
        alpha,
        config.cointegration.price_space,
    )

    if persist:
        meta = {
            "universe_reason": universe_reason,
            "symbols": list(frames.keys()),
            "n_symbols": len(frames),
            "tested_pairs": tested,
            "skipped_pairs": skipped,
            "n_significant": int(len(sig)),
            "pvalue_threshold": alpha,
            "min_obs": config.cointegration.min_obs,
            "price_space": config.cointegration.price_space,
        }
        save_scan_results(
            df,
            root=config.storage.root,
            exchange_id=eid,
            timeframe=config.ohlcv.timeframe,
            mode=config.cointegration.persist_mode,
            meta=meta,
            basename="scan_all",
        )
        save_scan_results(
            sig,
            root=config.storage.root,
            exchange_id=eid,
            timeframe=config.ohlcv.timeframe,
            mode=config.cointegration.persist_mode,
            meta=meta,
            basename="scan_significant",
        )

    return df, sig
