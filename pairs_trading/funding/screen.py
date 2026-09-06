"""Funding-rate screens (harvest candidates — not live entry rules)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pairs_trading.funding.config import FundingAppConfig, load_funding_config
from pairs_trading.funding.storage import FundingStore

logger = logging.getLogger(__name__)


def annualize_funding_rate(
    mean_rate_per_settlement: float,
    *,
    settlements_per_day: float = 3.0,
) -> float:
    """Simple annualization: mean_settlement_rate × settlements/day × 365."""
    return float(mean_rate_per_settlement) * float(settlements_per_day) * 365.0


def summarize_funding_series(
    df: pd.DataFrame,
    *,
    settlements_per_day: float = 3.0,
    recent_n: int = 21,
) -> dict:
    """Stats for one symbol's funding history."""
    if df.empty:
        return {
            "n_obs": 0,
            "mean_rate": float("nan"),
            "median_rate": float("nan"),
            "std_rate": float("nan"),
            "last_rate": float("nan"),
            "mean_abs_rate": float("nan"),
            "ann_mean": float("nan"),
            "ann_recent": float("nan"),
            "pct_positive": float("nan"),
            "recent_mean": float("nan"),
        }
    rates = df["funding_rate"].astype(float)
    recent = rates.tail(recent_n)
    return {
        "n_obs": int(len(rates)),
        "mean_rate": float(rates.mean()),
        "median_rate": float(rates.median()),
        "std_rate": float(rates.std(ddof=0)),
        "last_rate": float(rates.iloc[-1]),
        "mean_abs_rate": float(rates.abs().mean()),
        "ann_mean": annualize_funding_rate(float(rates.mean()), settlements_per_day=settlements_per_day),
        "ann_recent": annualize_funding_rate(
            float(recent.mean()), settlements_per_day=settlements_per_day
        ),
        "pct_positive": float((rates > 0).mean()),
        "recent_mean": float(recent.mean()),
    }


def screen_funding_store(
    config: FundingAppConfig | None = None,
    *,
    symbols: list[str] | None = None,
    exchange_id: str | None = None,
) -> pd.DataFrame:
    """Rank stored funding series by |ann_mean| / harvest attractiveness."""
    config = config or load_funding_config()
    eid = exchange_id or config.exchange.id
    store = FundingStore(config.storage.root, eid)
    stems = store.list_symbols()
    # Convert stems BTC_USDT_USDT → BTC/USDT:USDT heuristically
    available: list[str] = []
    if symbols:
        wanted = set(symbols)
        for stem in stems:
            # Prefer exact path load via provided symbols
            pass
        for sym in symbols:
            if store.exists(sym):
                available.append(sym)
    else:
        for stem in stems:
            # BTC_USDT_USDT.parquet from BTC/USDT:USDT
            parts = stem.split("_")
            if len(parts) >= 3 and parts[-1] == parts[-2]:
                # BASE/QUOTE:QUOTE
                available.append(f"{parts[0]}/{parts[1]}:{parts[2]}")
            elif len(parts) == 2:
                available.append(f"{parts[0]}/{parts[1]}")
            else:
                available.append(stem.replace("_", "/"))

    rows: list[dict] = []
    spd = config.funding.settlements_per_day
    for sym in available:
        df = store.load(sym)
        stats = summarize_funding_series(df, settlements_per_day=spd)
        rows.append({"symbol": sym, "exchange_id": eid, **stats})

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs_ann_mean"] = out["ann_mean"].abs()
    out = out.sort_values("abs_ann_mean", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    # Soft flag: abs mean rate above noise floor
    out["above_noise"] = out["mean_abs_rate"] >= config.funding.min_abs_rate
    return out
