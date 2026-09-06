"""Periodic collection loop (default: every 15 minutes)."""

from __future__ import annotations

import logging
import time
from typing import Callable

import pandas as pd

from funding_arb.collector import collect_once, connect_collector
from funding_arb.config import FundingArbAppConfig, load_funding_arb_config
from funding_arb.storage import save_per_symbol_ticks, save_snapshot_batch, snapshot_summary
from pairs_trading.data.exchange import ConnectedExchange

logger = logging.getLogger(__name__)


def run_collect_cycle(
    config: FundingArbAppConfig | None = None,
    *,
    symbols: list[str] | None = None,
    connected: ConnectedExchange | None = None,
    persist: bool = True,
    per_symbol_ticks: bool = True,
) -> tuple[pd.DataFrame, ConnectedExchange, list]:
    """Single collect → optional persist. Returns frame, connection, written paths."""
    config = config or load_funding_arb_config()
    frame, connected = collect_once(config, symbols=symbols, connected=connected)
    paths: list = []
    if persist and not frame.empty:
        paths = save_snapshot_batch(
            frame,
            root=config.storage.root,
            exchange_id=connected.exchange_id,
            mode=config.collector.persist_mode,
        )
        if per_symbol_ticks:
            paths.extend(
                save_per_symbol_ticks(
                    frame,
                    root=config.storage.root,
                    exchange_id=connected.exchange_id,
                )
            )
    return frame, connected, paths


def run_collect_loop(
    config: FundingArbAppConfig | None = None,
    *,
    symbols: list[str] | None = None,
    max_cycles: int | None = None,
    persist: bool = True,
    sleep_fn: Callable[[float], None] = time.sleep,
    on_cycle: Callable[[pd.DataFrame, dict], None] | None = None,
) -> int:
    """Block and refresh the local snapshot table every ``interval_seconds``.

    ``max_cycles=None`` runs forever. Returns number of completed cycles.
    """
    config = config or load_funding_arb_config()
    interval = float(config.collector.interval_seconds)
    connected = connect_collector(config)
    logger.info(
        "funding_arb collector loop: exchange=%s interval=%ss symbols=%d",
        connected.exchange_id,
        int(interval),
        len(symbols or config.collector.symbols),
    )
    cycles = 0
    while max_cycles is None or cycles < max_cycles:
        t0 = time.monotonic()
        try:
            frame, connected, paths = run_collect_cycle(
                config,
                symbols=symbols,
                connected=connected,
                persist=persist,
            )
            summary = snapshot_summary(frame)
            summary["paths"] = [str(p) for p in paths]
            logger.info("collect cycle %d: %s", cycles + 1, summary)
            if on_cycle is not None:
                on_cycle(frame, summary)
        except Exception:  # noqa: BLE001
            logger.exception("collect cycle %d failed", cycles + 1)
        cycles += 1
        if max_cycles is not None and cycles >= max_cycles:
            break
        elapsed = time.monotonic() - t0
        sleep_for = max(0.0, interval - elapsed)
        sleep_fn(sleep_for)
    return cycles
