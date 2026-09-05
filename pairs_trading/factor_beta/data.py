"""Step 1 — log-price panel via existing ParquetStore data layer."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pairs_trading.data.storage import ParquetStore
from pairs_trading.factor_beta.config import FactorBetaConfig, load_factor_beta_config


@dataclass(frozen=True)
class SymbolCoverage:
    symbol: str
    n_bars: int
    start: str | None
    end: str | None
    span_days: int
    on_disk: bool


@dataclass
class LogPricePanel:
    log_prices: pd.DataFrame
    factor_symbol: str
    exchange_id: str
    timeframe: str
    coverage: list[SymbolCoverage]
    missing_requested: list[str]
    rules_snapshot: dict


def list_on_disk_symbols(store: ParquetStore) -> list[str]:
    out: list[str] = []
    for stem in store.list_symbols():
        parts = stem.split("_")
        if len(parts) == 2:
            out.append(f"{parts[0]}/{parts[1]}")
        else:
            out.append(stem.replace("_", "/"))
    return sorted(out)


def coverage_for(store: ParquetStore, symbol: str) -> SymbolCoverage:
    if not store.exists(symbol):
        return SymbolCoverage(symbol, 0, None, None, 0, False)
    df = store.load(symbol)
    if df.empty:
        return SymbolCoverage(symbol, 0, None, None, 0, True)
    ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return SymbolCoverage(
        symbol=symbol,
        n_bars=int(len(df)),
        start=ts.min().date().isoformat(),
        end=ts.max().date().isoformat(),
        span_days=int((ts.max() - ts.min()).days),
        on_disk=True,
    )


def resolve_universe(
    config: FactorBetaConfig, store: ParquetStore
) -> tuple[list[str], list[str]]:
    requested = list(config.universe.symbols)
    on_disk = set(list_on_disk_symbols(store))
    missing = [s for s in requested if s not in on_disk]
    if config.universe.prefer_on_disk_intersection:
        selected = [s for s in requested if s in on_disk]
    else:
        selected = list(requested)
        if config.universe.require_on_disk and missing:
            raise FileNotFoundError(f"Missing parquet for: {missing}")
    factor = config.factor.symbol
    selected = [s for s in selected if s != factor][: config.universe.max_symbols]
    return selected, missing


def load_log_close(store: ParquetStore, symbol: str) -> pd.Series:
    df = store.load(symbol)
    if df.empty:
        raise FileNotFoundError(f"No OHLCV for {symbol}")
    close = df["close"].astype(float)
    if (close <= 0).any():
        raise ValueError(f"Non-positive close in {symbol}")
    ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    series = pd.Series(np.log(close.to_numpy()), index=ts, name=symbol)
    return series[~series.index.duplicated(keep="last")].sort_index()


def rules_snapshot(config: FactorBetaConfig) -> dict:
    return {
        "5a_costs_round_trip_bps": config.costs.round_trip_bps,
        "5a_costs_range_bps": [
            config.costs.round_trip_bps_min,
            config.costs.round_trip_bps_max,
        ],
        "5a_source": config.costs.source,
        "5b_min_edge_to_cost_multiple": config.entry_gate.min_edge_to_cost_multiple,
        "5c_min_independent_windows": config.evaluation.min_independent_windows,
        "5c_window_weeks": [
            config.evaluation.window_weeks_min,
            config.evaluation.window_weeks_max,
        ],
        "5c_non_overlapping": config.evaluation.windows_must_be_non_overlapping,
        "5d_dual_null": config.multiple_testing.dual_null,
        "5d_fdr_method": config.multiple_testing.fdr_method,
        "5d_fdr_q": config.multiple_testing.fdr_q,
        "5e_max_abs_residual_corr": config.common_factor_control.max_abs_residual_corr,
        "5e_action": config.common_factor_control.action,
        "stationarity_adf_pvalue_max": config.stationarity.adf_pvalue_max,
        "note": "z-entry NOT fixed — walk-forward owns it (step 6)",
    }


def build_log_price_panel(config: FactorBetaConfig | None = None) -> LogPricePanel:
    config = config or load_factor_beta_config()
    store = ParquetStore(
        Path(config.storage.root),
        config.storage.exchange_id,
        config.storage.timeframe,
    )
    selected, missing = resolve_universe(config, store)
    factor = config.factor.symbol
    if not store.exists(factor):
        raise FileNotFoundError(f"Factor {factor} missing under {store.base_dir}")

    series_map: dict[str, pd.Series] = {factor: load_log_close(store, factor)}
    for sym in selected:
        series_map[sym] = load_log_close(store, sym)

    aligned = pd.DataFrame(series_map).sort_index().dropna(how="any")
    coverage = [coverage_for(store, factor)] + [coverage_for(store, s) for s in selected]
    return LogPricePanel(
        log_prices=aligned,
        factor_symbol=factor,
        exchange_id=config.storage.exchange_id,
        timeframe=config.storage.timeframe,
        coverage=coverage,
        missing_requested=missing,
        rules_snapshot=rules_snapshot(config),
    )


def alignment_transparency(config: FactorBetaConfig | None = None) -> dict:
    """Per-coin RAW first/last bar vs aligned panel method (no forward-fill)."""
    config = config or load_factor_beta_config()
    store = ParquetStore(
        Path(config.storage.root),
        config.storage.exchange_id,
        config.storage.timeframe,
    )
    panel = build_log_price_panel(config)
    lp = panel.log_prices
    aligned_start = lp.index.min().date().isoformat() if len(lp) else None
    aligned_end = lp.index.max().date().isoformat() if len(lp) else None

    raw_rows: list[dict] = []
    for cov in panel.coverage:
        binds_alignment = bool(
            cov.start is not None and aligned_start is not None and cov.start == aligned_start
        )
        raw_rows.append(
            {
                **asdict(cov),
                "longer_than_aligned": bool(
                    cov.start is not None
                    and aligned_start is not None
                    and cov.start < aligned_start
                ),
                "binds_aligned_start": binds_alignment,
            }
        )

    binders = [r["symbol"] for r in raw_rows if r["binds_aligned_start"]]
    longer = [r["symbol"] for r in raw_rows if r["longer_than_aligned"]]
    return {
        "alignment_method": "inner_join_then_dropna_how_any",
        "forward_fill": False,
        "nan_fill": False,
        "aligned_start": aligned_start,
        "aligned_end": aligned_end,
        "aligned_bars": int(len(lp)),
        "symbols_binding_aligned_start": binders,
        "symbols_with_longer_raw_history": longer,
        "raw_per_symbol": raw_rows,
        "note": (
            "Aligned panel is truncated to the intersection of available dates. "
            "No forward-fill. Shorter raw history coins bind the panel start."
        ),
    }


def inventory_report(config: FactorBetaConfig | None = None) -> dict:
    config = config or load_factor_beta_config()
    panel = build_log_price_panel(config)
    lp = panel.log_prices
    tradeable = [c for c in lp.columns if c != panel.factor_symbol]
    span_days = int((lp.index.max() - lp.index.min()).days) if len(lp) >= 2 else 0
    weeks = span_days / 7.0
    min_weeks_needed = (
        config.evaluation.min_independent_windows * config.evaluation.window_weeks_min
    )
    align = alignment_transparency(config)
    return {
        "step": 1,
        "step_name": "data",
        "status": "complete",
        "exchange_id": panel.exchange_id,
        "timeframe": panel.timeframe,
        "factor_symbol": panel.factor_symbol,
        "n_tradeable": len(tradeable),
        "tradeable_symbols": tradeable,
        "missing_requested": panel.missing_requested,
        "aligned_bars": int(len(lp)),
        "aligned_start": lp.index.min().date().isoformat() if len(lp) else None,
        "aligned_end": lp.index.max().date().isoformat() if len(lp) else None,
        "aligned_span_days": span_days,
        "aligned_span_weeks": round(weeks, 2),
        "rule5c_min_weeks_needed": min_weeks_needed,
        "rule5c_calendar_feasible": weeks >= min_weeks_needed,
        "alignment": align,
        "coverage": [asdict(c) for c in panel.coverage],
        "rules_locked": panel.rules_snapshot,
        "interpretation_allowed": False,
        "next_step": 2,
        "next_step_name": "factor_model_rolling_beta",
        "note": (
            "Step 1 only verifies data + Rule-5 lock-in. "
            "No alpha claim, no signal, no 'good news'."
        ),
    }


def save_inventory(report: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "step1_data_inventory.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
