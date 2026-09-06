"""BNB (or any symbol) hub diagnostic: placebo cointegration checks."""

from __future__ import annotations

import logging
from dataclasses import asdict

import pandas as pd

from pairs_trading.cointegration.engine import evaluate_pair
from pairs_trading.cointegration.fdr import apply_benjamini_hochberg, hub_involvement_summary
from pairs_trading.data.config import DataConfig
from pairs_trading.data.storage import ParquetStore

logger = logging.getLogger(__name__)

# Economically distant / narrative-weak placebos relative to an exchange token.
DEFAULT_PLACEBOS = (
    "XAUT/USDT",  # gold
    "PEPE/USDT",  # meme
    "DOGE/USDT",  # meme
    "LTC/USDT",  # legacy L1, weak BNB narrative
    "ATOM/USDT",  # cosmos
    "NEAR/USDT",
    "APT/USDT",
    "OP/USDT",  # L2, not BNB-chain
)


def run_hub_placebo_tests(
    config: DataConfig,
    *,
    hub_symbol: str = "BNB/USDT",
    placebos: list[str] | None = None,
    exchange_id: str | None = None,
) -> pd.DataFrame:
    """Evaluate hub vs each placebo on log prices; return ranked table + FDR."""
    from pairs_trading.cointegration.scanner import resolve_exchange_id

    eid = resolve_exchange_id(config, exchange_id)
    store = ParquetStore(config.storage.root, eid, config.ohlcv.timeframe)
    hub_df = store.load(hub_symbol)
    if hub_df.empty:
        raise FileNotFoundError(f"No OHLCV for hub {hub_symbol} under {store.base_dir}")

    placebos = list(placebos or DEFAULT_PLACEBOS)
    rows: list[dict] = []
    for sym in placebos:
        if sym == hub_symbol:
            continue
        other = store.load(sym)
        if other.empty:
            logger.warning("Placebo %s missing — skip", sym)
            continue
        try:
            hit = evaluate_pair(
                hub_symbol,
                sym,
                hub_df,
                other,
                min_obs=config.cointegration.min_obs,
                price_space=config.cointegration.price_space,
            )
        except ValueError as exc:
            logger.warning("Placebo %s failed: %s", sym, exc)
            continue
        row = asdict(hit)
        row["pair"] = f"{hit.y_symbol}|{hit.x_symbol}"
        row["placebo"] = True
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    alpha = config.cointegration.pvalue_threshold
    df = df.sort_values("pvalue").reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return apply_benjamini_hochberg(df, pvalue_col="pvalue", alpha=alpha)


def summarize_hub_from_scan(
    scan_df: pd.DataFrame,
    *,
    hub_symbol: str = "BNB/USDT",
) -> dict:
    """Hub hit-rate on an existing full-universe scan table."""
    raw = hub_involvement_summary(scan_df, hub_symbol=hub_symbol, significant_col="significant_raw")
    fdr = hub_involvement_summary(scan_df, hub_symbol=hub_symbol, significant_col="significant_fdr")
    return {"raw": raw, "fdr": fdr}
