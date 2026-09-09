"""Pairwise cointegration scan over a point-in-time universe."""

from __future__ import annotations

import itertools
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pairs_trading.cointegration.engine import CointegrationHit, evaluate_pair
from pairs_trading.cointegration.fdr import apply_benjamini_hochberg, hub_involvement_summary
from pairs_trading.cointegration.neutralize import build_factor_residual_frames
from pairs_trading.cointegration.persist import save_scan_results
from pairs_trading.data.config import DataConfig, load_config
from pairs_trading.data.storage import ParquetStore
from pairs_trading.data.universe import load_universe_snapshot

logger = logging.getLogger(__name__)

__all__ = [
    "CointegrationHit",
    "resolve_universe_symbols",
    "scan_pairs",
    "hub_involvement_summary",
]


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
    preprocess: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test all unordered pairs; return (full_ranked_df, significant_fdr_df).

    preprocess:
      - none: Engle-Granger on configured price_space (default log)
      - btc_residual: residualize each coin vs factor_symbol, then EG on residuals
        (factor symbol excluded from pairs; residuals evaluated as level series)
    """
    config = config or load_config()
    eid = resolve_exchange_id(config, exchange_id)
    store = ParquetStore(config.storage.root, eid, config.ohlcv.timeframe)
    alpha = config.cointegration.pvalue_threshold
    preprocess = preprocess or config.cointegration.preprocess
    factor_symbol = config.cointegration.factor_symbol

    universe, universe_reason = resolve_universe_symbols(
        config, store, symbols=symbols, universe_as_of=universe_as_of
    )
    # Residual mode needs the factor series even if it is outside the snapshot list.
    load_syms = list(universe)
    if preprocess == "btc_residual" and factor_symbol not in load_syms:
        load_syms.append(factor_symbol)

    logger.info(
        "Cointegration scan on %s (%s, price_space=%s, preprocess=%s) — %d symbols via %s",
        eid,
        config.ohlcv.timeframe,
        config.cointegration.price_space,
        preprocess,
        len(universe),
        universe_reason,
    )

    frames: dict[str, pd.DataFrame] = {}
    for sym in load_syms:
        try:
            frames[sym] = load_symbol_frame(store, sym)
        except FileNotFoundError:
            logger.warning("Skipping %s — no local OHLCV", sym)

    eval_price_space = config.cointegration.price_space
    factor_meta: dict[str, dict[str, float | int]] = {}
    if preprocess == "btc_residual":
        frames, factor_meta = build_factor_residual_frames(
            frames,
            factor_symbol=factor_symbol,
            min_obs=config.cointegration.min_obs,
        )
        # Residuals are already log-space innovations — do not log again.
        eval_price_space = "level"
        logger.info(
            "Factor-neutral frames ready: %d symbols (excluded factor %s)",
            len(frames),
            factor_symbol,
        )
    elif preprocess != "none":
        raise ValueError(f"Unknown preprocess={preprocess!r}")

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
                price_space=eval_price_space,
            )
        except ValueError as exc:
            skipped += 1
            logger.debug("Skip %s-%s: %s", y_sym, x_sym, exc)
            continue
        row = asdict(hit)
        row["pair"] = f"{hit.y_symbol}|{hit.x_symbol}"
        row["preprocess"] = preprocess
        row["eval_price_space"] = eval_price_space
        # Surface configured intent distinctly from internal eval transform.
        row["price_space"] = (
            "btc_residual" if preprocess == "btc_residual" else hit.price_space
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("pvalue", kind="mergesort").reset_index(drop=True)
        df.insert(0, "rank", range(1, len(df) + 1))
        df = apply_benjamini_hochberg(df, pvalue_col="pvalue", alpha=alpha)
        df = df.sort_values(["pvalue_fdr", "pvalue"], kind="mergesort").reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)

    sig_raw = df[df["significant_raw"]].copy() if not df.empty else df.copy()
    sig_fdr = df[df["significant_fdr"]].copy() if not df.empty else df.copy()
    sig = sig_fdr

    logger.info(
        "Scan complete: tested=%d skipped=%d raw_p<%.3f=%d fdr_p<%.3f=%d "
        "(preprocess=%s)",
        tested,
        skipped,
        alpha,
        len(sig_raw),
        len(sig_fdr),
        preprocess,
    )

    basename_suffix = "residual" if preprocess == "btc_residual" else "log"
    if persist:
        meta = {
            "universe_reason": universe_reason,
            "symbols": list(frames.keys()),
            "n_symbols": len(frames),
            "tested_pairs": tested,
            "skipped_pairs": skipped,
            "n_significant_raw": int(len(sig_raw)),
            "n_significant_fdr": int(len(sig_fdr)),
            "pvalue_threshold": alpha,
            "fdr_method": "benjamini_hochberg",
            "min_obs": config.cointegration.min_obs,
            "price_space": config.cointegration.price_space,
            "preprocess": preprocess,
            "factor_symbol": factor_symbol if preprocess == "btc_residual" else None,
            "factor_betas": factor_meta if preprocess == "btc_residual" else None,
        }
        save_scan_results(
            df,
            root=config.storage.root,
            exchange_id=eid,
            timeframe=config.ohlcv.timeframe,
            mode=config.cointegration.persist_mode,
            meta=meta,
            basename=f"scan_all_{basename_suffix}",
        )
        save_scan_results(
            sig_raw,
            root=config.storage.root,
            exchange_id=eid,
            timeframe=config.ohlcv.timeframe,
            mode=config.cointegration.persist_mode,
            meta=meta,
            basename=f"scan_significant_raw_{basename_suffix}",
        )
        save_scan_results(
            sig_fdr,
            root=config.storage.root,
            exchange_id=eid,
            timeframe=config.ohlcv.timeframe,
            mode=config.cointegration.persist_mode,
            meta=meta,
            basename=f"scan_significant_fdr_{basename_suffix}",
        )

    return df, sig
