"""End-to-end Module 1 harvest: funding + basis + fees + persistence + capacity + size + paper."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from pairs_trading.funding.basis import fetch_basis_snapshot
from pairs_trading.funding.capacity import CapacityConfig, apply_capacity_filter
from pairs_trading.funding.config import FundingAppConfig, load_funding_config
from pairs_trading.funding.fees import FeeConfig, fee_drag_annualized, round_trip_cost
from pairs_trading.funding.paper import PaperResult, paper_results_frame, simulate_funding_harvest
from pairs_trading.funding.persistence import funding_persistence
from pairs_trading.funding.risk import RiskConfig, size_position
from pairs_trading.funding.screen import annualize_funding_rate, summarize_funding_series
from pairs_trading.funding.storage import FundingStore

logger = logging.getLogger(__name__)


def list_stored_swap_symbols(store: FundingStore) -> list[str]:
    out: list[str] = []
    for stem in store.list_symbols():
        parts = stem.split("_")
        if len(parts) >= 3 and parts[-1] == parts[-2]:
            out.append(f"{parts[0]}/{parts[1]}:{parts[2]}")
        elif len(parts) == 2:
            out.append(f"{parts[0]}/{parts[1]}")
        else:
            out.append(stem.replace("_", "/"))
    return out


def build_harvest_table(
    config: FundingAppConfig | None = None,
    *,
    exchange: Any | None = None,
    exchange_id: str | None = None,
    symbols: list[str] | None = None,
    fee_config: FeeConfig | None = None,
    capacity_config: CapacityConfig | None = None,
    risk_config: RiskConfig | None = None,
) -> pd.DataFrame:
    """Rank symbols by expected net annualized carry after fees & persistence."""
    config = config or load_funding_config()
    fee_config = fee_config or config.fees
    capacity_config = capacity_config or config.capacity
    risk_config = risk_config or config.risk
    eid = exchange_id or config.exchange.id
    store = FundingStore(config.storage.root, eid)

    if symbols:
        available = [s for s in symbols if store.exists(s)]
    else:
        available = list_stored_swap_symbols(store)
        if not available:
            available = [s for s in config.funding.default_symbols if store.exists(s)]

    spd = config.funding.settlements_per_day
    hold_days = config.harvest.expected_hold_days
    fee_drag = fee_drag_annualized(fee_config, hold_days=hold_days)
    rt = round_trip_cost(fee_config)

    rows: list[dict] = []
    for sym in available:
        hist = store.load(sym)
        stats = summarize_funding_series(hist, settlements_per_day=spd)
        pers = (
            funding_persistence(hist["funding_rate"])
            if not hist.empty
            else {
                "sign_persistence": float("nan"),
                "autocorr_lag1": float("nan"),
                "same_sign_streak_mean": float("nan"),
                "harvest_efficiency": float("nan"),
            }
        )

        basis: dict = {}
        if exchange is not None:
            try:
                basis = fetch_basis_snapshot(exchange, sym)
            except Exception as exc:  # noqa: BLE001
                logger.warning("basis snapshot failed %s: %s", sym, exc)

        mean_rate = float(stats["mean_rate"]) if stats["n_obs"] else float("nan")
        eff = pers.get("harvest_efficiency", float("nan"))
        if eff != eff:  # NaN
            eff = 1.0
        if not config.harvest.apply_persistence:
            eff = 1.0

        if mean_rate != mean_rate:
            gross_ann = float("nan")
        else:
            gross_ann = annualize_funding_rate(
                abs(mean_rate) * float(eff), settlements_per_day=spd
            )

        basis_frac = basis.get("basis_mark_index")
        if basis_frac is None:
            basis_frac = basis.get("basis_mark_spot")
        if (
            not config.harvest.apply_basis_drag
            or basis_frac is None
            or basis_frac != basis_frac
        ):
            basis_drag_ann = 0.0
            basis_bps = float("nan")
        else:
            basis_drag_ann = abs(float(basis_frac)) * (365.0 / hold_days)
            basis_bps = float(basis_frac) * 10_000.0

        net_ann = (
            float("nan")
            if gross_ann != gross_ann
            else float(gross_ann) - fee_drag - basis_drag_ann
        )

        qv = basis.get("quote_volume_24h")
        sized = size_position(
            sym,
            funding_rate_signal=0.0 if mean_rate != mean_rate else mean_rate,
            quote_volume_24h=float(qv) if qv is not None else None,
            risk=risk_config,
            capacity=capacity_config,
        )

        rows.append(
            {
                "symbol": sym,
                "exchange_id": eid,
                **stats,
                "sign_persistence": pers.get("sign_persistence"),
                "autocorr_lag1": pers.get("autocorr_lag1"),
                "same_sign_streak_mean": pers.get("same_sign_streak_mean"),
                "harvest_efficiency": pers.get("harvest_efficiency"),
                "basis_bps": basis_bps,
                "quote_volume_24h": qv,
                "gross_ann": gross_ann,
                "fee_drag_ann": fee_drag,
                "basis_drag_ann": basis_drag_ann,
                "net_ann": net_ann,
                "round_trip_fee": rt,
                "side": sized.side,
                "sized_notional": sized.notional_usdt,
                "size_reason": sized.reason,
                "above_noise": float(stats.get("mean_abs_rate") or 0)
                >= config.funding.min_abs_rate,
                "positive_net": (net_ann == net_ann) and (net_ann > config.harvest.min_net_ann),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = apply_capacity_filter(out, capacity_config)
    out["tradable"] = (
        out["passes_capacity"]
        & out["positive_net"]
        & (out["sized_notional"] > 0)
        & out["above_noise"]
    )
    out = out.sort_values("net_ann", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


def run_paper_backtest(
    config: FundingAppConfig | None = None,
    *,
    exchange_id: str | None = None,
    harvest_df: pd.DataFrame | None = None,
    fee_config: FeeConfig | None = None,
    exchange: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (harvest_table, paper_results). Builds harvest if not provided."""
    config = config or load_funding_config()
    fee_config = fee_config or config.fees
    eid = exchange_id or config.exchange.id
    store = FundingStore(config.storage.root, eid)
    table = (
        harvest_df
        if harvest_df is not None
        else build_harvest_table(config, exchange=exchange, exchange_id=eid, fee_config=fee_config)
    )
    results: list[PaperResult] = []
    for _, row in table.iterrows():
        sym = str(row["symbol"])
        notional = float(row.get("sized_notional") or 0.0)
        if notional <= 0:
            notional = float(config.risk.equity_usdt * config.risk.target_gross_fraction)
        hist = store.load(sym)
        if hist.empty:
            continue
        entry_b = 0.0
        exit_b = 0.0
        bps = row.get("basis_bps")
        if bps is not None and bps == bps:
            entry_b = float(bps) / 10_000.0
            exit_b = entry_b * 0.5
        mean_rate = row.get("mean_rate")
        res = simulate_funding_harvest(
            hist["funding_rate"],
            symbol=sym,
            notional_usdt=notional,
            fee_config=fee_config,
            settlements_per_day=config.funding.settlements_per_day,
            side_rate=float(mean_rate) if mean_rate == mean_rate else None,
            entry_basis=entry_b,
            exit_basis=exit_b,
        )
        results.append(res)
    return table, paper_results_frame(results)
