"""Paper simulator: GBM mid + queue-intensity fills + fees + adverse selection."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from market_making.config import MarketMakingConfig
from market_making.engine import QuoteEngine
from market_making.fees import fee_on_notional
from market_making.inventory import InventoryState


@dataclass
class SimResult:
    n_bars: int
    n_buys: int
    n_sells: int
    fees_paid: float
    realized_pnl: float
    start_equity: float
    final_equity: float
    final_inventory: float
    total_return_pct: float
    net_positive: bool
    avg_half_spread_bps: float
    required_half_spread_bps: float
    commission_pct: float
    commission_bps: float
    interpretation_allowed: bool
    notes: str


def _gbm_path(
    n: int, mid0: float, ann_vol: float, bars_per_year: float, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dt = 1.0 / bars_per_year
    shocks = rng.normal(0.0, ann_vol * np.sqrt(dt), size=n)
    return mid0 * np.exp(np.cumsum(shocks))


def _fill_prob(mid: float, quote_px: float, half_bps: float, base_p: float) -> float:
    """Maker-queue intensity: higher when mid is near the quote."""
    half = max(half_bps / 1e4, 1e-8)
    gap = abs(mid - quote_px) / mid
    return float(base_p * np.exp(-gap / half))


def run_paper_sim(cfg: MarketMakingConfig) -> tuple[SimResult, pd.DataFrame]:
    sc = cfg.sim
    mids = _gbm_path(sc.n_bars, sc.mid0, sc.ann_vol, sc.bars_per_year, sc.seed)
    eng = QuoteEngine(cfg)
    inv = InventoryState(
        base=0.0,
        quote=cfg.max_inventory * sc.mid0 * 2.0,
        target=cfg.target_inventory,
        max_abs=cfg.max_inventory,
        halt_frac=cfg.inventory_halt_frac,
    )
    start_eq = inv.mark_to_market(float(mids[0]))
    rng = np.random.default_rng(sc.seed + 17)
    rows: list[dict[str, Any]] = []
    halfs: list[float] = []

    for i, mid in enumerate(mids):
        mid_f = float(mid)
        q = eng.quote(mid_f, inv, mid_history=mids[: i + 1])
        if q.quoting:
            halfs.append(q.half_spread_bps)

        buy_fill = False
        sell_fill = False
        if q.quoting and q.bid_size > 0:
            p_bid = _fill_prob(mid_f, q.bid, q.half_spread_bps, sc.fill_prob)
            if rng.random() < p_bid:
                # Adverse selection: buy fill slightly above bid.
                px = q.bid * (1.0 + sc.adverse_frac * (q.half_spread_bps / 1e4))
                fee = fee_on_notional(px * q.bid_size, cfg)
                inv.apply_buy(price=px, size=q.bid_size, fee=fee, bar=i)
                buy_fill = True
        if q.quoting and q.ask_size > 0:
            p_ask = _fill_prob(mid_f, q.ask, q.half_spread_bps, sc.fill_prob)
            if rng.random() < p_ask:
                px = q.ask * (1.0 - sc.adverse_frac * (q.half_spread_bps / 1e4))
                fee = fee_on_notional(px * q.ask_size, cfg)
                inv.apply_sell(price=px, size=q.ask_size, fee=fee, bar=i)
                sell_fill = True

        rows.append(
            {
                "bar": i,
                "mid": mid_f,
                "bid": q.bid if q.quoting else np.nan,
                "ask": q.ask if q.quoting else np.nan,
                "half_spread_bps": q.half_spread_bps if q.quoting else np.nan,
                "net_edge_bps": q.net_edge_bps_if_matched if q.quoting else np.nan,
                "quoting": q.quoting,
                "reason": q.reason,
                "inventory": inv.base,
                "equity": inv.mark_to_market(mid_f),
                "buy_fill": buy_fill,
                "sell_fill": sell_fill,
                "fees_cum": inv.fees_paid,
                "realized_pnl": inv.realized_pnl,
            }
        )

    final_eq = inv.mark_to_market(float(mids[-1]))
    ret = 100.0 * (final_eq - start_eq) / start_eq if start_eq else 0.0
    trades = inv.n_buys + inv.n_sells
    result = SimResult(
        n_bars=sc.n_bars,
        n_buys=inv.n_buys,
        n_sells=inv.n_sells,
        fees_paid=float(inv.fees_paid),
        realized_pnl=float(inv.realized_pnl),
        start_equity=float(start_eq),
        final_equity=float(final_eq),
        final_inventory=float(inv.base),
        total_return_pct=float(ret),
        net_positive=bool(final_eq > start_eq and trades > 0),
        avg_half_spread_bps=float(np.mean(halfs)) if halfs else 0.0,
        required_half_spread_bps=float(cfg.required_half_spread_bps),
        commission_pct=float(cfg.commission_pct),
        commission_bps=float(cfg.commission_bps),
        interpretation_allowed=False,
        notes=(
            f"Paper GBM+queue fills. Gate half-spread>="
            f"{cfg.required_half_spread_bps:.1f}bps "
            f"(fee {cfg.commission_bps:.1f}bps + buffer {cfg.edge_buffer_bps:.1f}bps). "
            "Not a live-edge claim."
        ),
    )
    return result, pd.DataFrame(rows)


def save_sim_artifacts(
    result: SimResult,
    frame: pd.DataFrame,
    *,
    root: str | Path | None = None,
    exchange_id: str = "okx",
) -> dict[str, str]:
    out = Path(root or "data") / exchange_id / "market_making"
    out.mkdir(parents=True, exist_ok=True)
    summary = out / "paper_sim_latest.json"
    parquet = out / "paper_sim_latest.parquet"
    csv = out / "paper_sim_latest.csv"
    summary.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    frame.to_parquet(parquet, index=False)
    frame.to_csv(csv, index=False)
    return {"summary": str(summary), "parquet": str(parquet), "csv": str(csv)}
