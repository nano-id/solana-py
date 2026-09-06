"""Multi-slot single-wallet MM portfolio.

Rules
-----
1. Shared USDT wallet.
2. Up to ``max_slots`` concurrent open symbols (default 3).
3. Each new slot armed with fixed budget ``slot_notional_usd`` (default 3500):
   max_inventory ≈ slot_notional / mid.
4. When free slots exist, open best unused signals first
   (score = net_edge_bps_if_matched).
5. Each open slot managed until flat (per-slot lock).

Paper fill model matches the single-symbol simulator. Not a live-edge claim.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from market_making.config import MarketMakingConfig
from market_making.engine import Quote, QuoteEngine
from market_making.fees import fee_on_notional
from market_making.inventory import InventoryState
from market_making.portfolio import load_aligned_panel
from market_making.simulator import _fill_prob


@dataclass
class SlotState:
    symbol: str
    inv: InventoryState
    eng: QuoteEngine
    notional_budget: float


@dataclass
class MultiPortfolioResult:
    n_bars: int
    n_entries: int
    n_round_trips: int
    n_buys: int
    n_sells: int
    n_skipped_busy: int
    n_multi_signal_choices: int
    max_slots: int
    slot_notional_usd: float
    fees_paid: float
    realized_pnl: float
    start_equity: float
    final_equity: float
    total_return_pct: float
    net_positive: bool
    active_symbol_bars: dict[str, int]
    symbols: list[str]
    commission_pct: float
    interpretation_allowed: bool
    notes: str
    open_slots_end: list[str] = field(default_factory=list)


def _score(q: Quote) -> float:
    if not q.quoting:
        return float("-inf")
    return float(q.net_edge_bps_if_matched)


def run_multi_portfolio_panel(
    cfg: MarketMakingConfig,
    panel: dict[str, np.ndarray],
    *,
    timestamps: pd.DatetimeIndex | np.ndarray | None = None,
    seed: int = 42,
    initial_equity: float = 10_500.0,
    max_slots: int = 3,
    slot_notional_usd: float = 3_500.0,
    notes: str | None = None,
) -> tuple[MultiPortfolioResult, pd.DataFrame]:
    symbols = list(panel.keys())
    if not symbols:
        raise ValueError("Empty panel")
    n = len(next(iter(panel.values())))
    if any(len(v) != n for v in panel.values()):
        raise ValueError("Panel length mismatch")
    if n < 2:
        raise ValueError("Need >= 2 bars")
    if max_slots < 1:
        raise ValueError("max_slots must be >= 1")
    if slot_notional_usd <= 0 or initial_equity <= 0:
        raise ValueError("equity/slot_notional must be > 0")

    flat_eps = float(cfg.portfolio_flat_eps)
    sc = cfg.sim
    rng = np.random.default_rng(seed + 77)

    cash = float(initial_equity)
    slots: dict[str, SlotState] = {}
    hist: dict[str, list[float]] = {s: [] for s in symbols}

    n_entries = 0
    n_round_trips = 0
    n_buys = 0
    n_sells = 0
    n_skipped_busy = 0
    n_multi = 0
    fees_total = 0.0
    realized_total = 0.0
    active_bars: dict[str, int] = {s: 0 for s in symbols}
    rows: list[dict[str, Any]] = []

    def mark_equity(mids_now: dict[str, float]) -> float:
        eq = cash
        for sym, slot in slots.items():
            eq += slot.inv.base * mids_now[sym]
        return float(eq)

    def arm(sym: str, mid: float) -> SlotState:
        max_inv = max(slot_notional_usd / mid, 1e-12)
        inv = InventoryState(
            base=0.0,
            quote=0.0,
            target=0.0,
            max_abs=max_inv,
            halt_frac=cfg.inventory_halt_frac,
        )
        eng = QuoteEngine(cfg.model_copy(update={"max_inventory": max_inv}))
        return SlotState(
            symbol=sym, inv=inv, eng=eng, notional_budget=slot_notional_usd
        )

    for i in range(n):
        mids_now = {s: float(panel[s][i]) for s in symbols}
        for s in symbols:
            hist[s].append(mids_now[s])
        ts_val = timestamps[i] if timestamps is not None else None

        closed_this_bar: list[str] = []
        opened_this_bar: list[str] = []
        n_cand = 0

        # Manage open slots
        for sym in list(slots.keys()):
            active_bars[sym] += 1
            slot = slots[sym]
            mid = mids_now[sym]
            q = slot.eng.quote(
                mid, slot.inv, mid_history=np.asarray(hist[sym], dtype=float)
            )
            if q.quoting:
                if q.bid_size > 0 and rng.random() < _fill_prob(
                    mid, q.bid, q.half_spread_bps, sc.fill_prob
                ):
                    px = q.bid * (1.0 + sc.adverse_frac * (q.half_spread_bps / 1e4))
                    fee = fee_on_notional(px * q.bid_size, cfg)
                    cost = px * q.bid_size + fee
                    if cost <= cash + 1e-9 and (
                        slot.inv.base * mid + cost <= slot_notional_usd * 1.05 + 1e-9
                    ):
                        cash -= cost
                        prev = slot.inv.realized_pnl
                        slot.inv.apply_buy(price=px, size=q.bid_size, fee=0.0, bar=i)
                        fees_total += fee
                        realized_total += slot.inv.realized_pnl - prev
                        n_buys += 1

                if q.ask_size > 0 and rng.random() < _fill_prob(
                    mid, q.ask, q.half_spread_bps, sc.fill_prob
                ):
                    sell_sz = min(q.ask_size, max(slot.inv.base, 0.0))
                    if sell_sz > flat_eps:
                        px = q.ask * (
                            1.0 - sc.adverse_frac * (q.half_spread_bps / 1e4)
                        )
                        fee = fee_on_notional(px * sell_sz, cfg)
                        cash += px * sell_sz - fee
                        prev = slot.inv.realized_pnl
                        slot.inv.apply_sell(price=px, size=sell_sz, fee=0.0, bar=i)
                        fees_total += fee
                        realized_total += slot.inv.realized_pnl - prev
                        n_sells += 1

            if abs(slot.inv.base) <= flat_eps:
                if abs(slot.inv.base) > 0:
                    cash += slot.inv.base * mid
                    slot.inv.base = 0.0
                n_round_trips += 1
                closed_this_bar.append(sym)
                del slots[sym]

        # Busy opportunity cost when full
        if len(slots) >= max_slots:
            for s in symbols:
                if s in slots:
                    continue
                mid_s = mids_now[s]
                if mid_s <= 0 or not np.isfinite(mid_s):
                    continue
                max_inv = max(slot_notional_usd / mid_s, 1e-12)
                probe = QuoteEngine(cfg.model_copy(update={"max_inventory": max_inv}))
                flat_inv = InventoryState(
                    base=0.0,
                    quote=0.0,
                    target=0.0,
                    max_abs=max_inv,
                    halt_frac=cfg.inventory_halt_frac,
                )
                q_s = probe.quote(
                    mid_s, flat_inv, mid_history=np.asarray(hist[s], dtype=float)
                )
                if not q_s.quoting or q_s.net_edge_bps_if_matched <= 0:
                    continue
                if rng.random() < _fill_prob(
                    mid_s, q_s.bid, q_s.half_spread_bps, sc.fill_prob
                ):
                    n_skipped_busy += 1

        # Open new slots
        free_slots = max_slots - len(slots)
        if free_slots > 0:
            candidates: list[tuple[str, float, Quote]] = []
            for s in symbols:
                if s in slots:
                    continue
                mid = mids_now[s]
                if mid <= 0 or not np.isfinite(mid):
                    continue
                max_inv = max(slot_notional_usd / mid, 1e-12)
                local = QuoteEngine(cfg.model_copy(update={"max_inventory": max_inv}))
                flat_inv = InventoryState(
                    base=0.0,
                    quote=0.0,
                    target=0.0,
                    max_abs=max_inv,
                    halt_frac=cfg.inventory_halt_frac,
                )
                q = local.quote(
                    mid, flat_inv, mid_history=np.asarray(hist[s], dtype=float)
                )
                if not q.quoting or q.net_edge_bps_if_matched <= 0 or q.bid_size <= 0:
                    continue
                if rng.random() < _fill_prob(
                    mid, q.bid, q.half_spread_bps, sc.fill_prob
                ):
                    candidates.append((s, _score(q), q))
            n_cand = len(candidates)
            if n_cand > 1:
                n_multi += 1
            candidates.sort(key=lambda t: t[1], reverse=True)
            for sym, _sc0, q0 in candidates:
                if free_slots <= 0:
                    break
                mid = mids_now[sym]
                px = q0.bid * (1.0 + sc.adverse_frac * (q0.half_spread_bps / 1e4))
                fee = fee_on_notional(px * q0.bid_size, cfg)
                cost = px * q0.bid_size + fee
                if cost > cash + 1e-9:
                    continue
                slot = arm(sym, mid)
                cash -= cost
                prev = slot.inv.realized_pnl
                slot.inv.apply_buy(price=px, size=q0.bid_size, fee=0.0, bar=i)
                fees_total += fee
                realized_total += slot.inv.realized_pnl - prev
                slots[sym] = slot
                n_entries += 1
                n_buys += 1
                opened_this_bar.append(sym)
                free_slots -= 1

        rows.append(
            {
                "bar": i,
                "timestamp": ts_val,
                "cash": cash,
                "equity": mark_equity(mids_now),
                "n_open_slots": len(slots),
                "open_symbols": ",".join(sorted(slots.keys())),
                "opened": ",".join(opened_this_bar),
                "closed": ",".join(closed_this_bar),
                "n_candidates": n_cand,
                "fees_cum": fees_total,
                "realized_pnl": realized_total,
            }
        )

    # Flatten leftovers
    if slots:
        last_mids = {s: float(panel[s][-1]) for s in symbols}
        for sym, slot in list(slots.items()):
            cash += slot.inv.base * last_mids[sym]
            slot.inv.base = 0.0
            n_round_trips += 1
            del slots[sym]

    final_eq = float(cash)
    ret = (
        100.0 * (final_eq - initial_equity) / initial_equity if initial_equity else 0.0
    )
    default_notes = (
        f"Multi-slot MM max_slots={max_slots} slot_notional=${slot_notional_usd:.0f}. "
        f"Fee {cfg.commission_pct}%/side. Not a live-edge claim."
    )
    result = MultiPortfolioResult(
        n_bars=n,
        n_entries=n_entries,
        n_round_trips=n_round_trips,
        n_buys=n_buys,
        n_sells=n_sells,
        n_skipped_busy=n_skipped_busy,
        n_multi_signal_choices=n_multi,
        max_slots=max_slots,
        slot_notional_usd=float(slot_notional_usd),
        fees_paid=float(fees_total),
        realized_pnl=float(realized_total),
        start_equity=float(initial_equity),
        final_equity=final_eq,
        total_return_pct=float(ret),
        net_positive=bool(final_eq > initial_equity and n_entries > 0),
        active_symbol_bars=active_bars,
        symbols=symbols,
        commission_pct=float(cfg.commission_pct),
        interpretation_allowed=False,
        notes=notes or default_notes,
        open_slots_end=sorted(slots.keys()),
    )
    return result, pd.DataFrame(rows)


def run_multi_portfolio_range(
    cfg: MarketMakingConfig,
    *,
    start: str,
    end_exclusive: str,
    timeframe: str = "1h",
    seed: int = 42,
    initial_equity: float = 10_500.0,
    max_slots: int = 3,
    slot_notional_usd: float = 3_500.0,
    symbols: list[str] | None = None,
) -> tuple[MultiPortfolioResult, pd.DataFrame]:
    syms = list(symbols or cfg.symbols)
    stamps, panel = load_aligned_panel(
        syms,
        timeframe=timeframe,
        start=start,
        end_exclusive=end_exclusive,
        root=cfg.storage_root,
        exchange_id=cfg.exchange_id,
    )
    notes = (
        f"Multi-slot {start}→{end_exclusive} {timeframe} on {cfg.exchange_id}. "
        f"n_symbols={len(syms)} n_bars={len(stamps)} max_slots={max_slots} "
        f"slot=${slot_notional_usd:.0f} equity0=${initial_equity:.0f}. "
        f"Fee {cfg.commission_pct}%/side. Not a live-edge claim."
    )
    return run_multi_portfolio_panel(
        cfg,
        panel,
        timestamps=stamps,
        seed=seed,
        initial_equity=initial_equity,
        max_slots=max_slots,
        slot_notional_usd=slot_notional_usd,
        notes=notes,
    )


def save_multi_portfolio_artifacts(
    result: MultiPortfolioResult,
    frame: pd.DataFrame,
    *,
    root: str | Path | None = None,
    exchange_id: str = "okx",
    basename: str = "multi_portfolio_latest",
) -> dict[str, str]:
    out = Path(root or "data") / exchange_id / "market_making"
    out.mkdir(parents=True, exist_ok=True)
    summary = out / f"{basename}.json"
    csv = out / f"{basename}.csv"
    summary.write_text(
        json.dumps(asdict(result), indent=2, default=str), encoding="utf-8"
    )
    frame.to_csv(csv, index=False)
    art = Path("/opt/cursor/artifacts")
    art.mkdir(parents=True, exist_ok=True)
    art_json = art / f"{basename}.json"
    art_csv = art / f"{basename}.csv"
    art_json.write_text(summary.read_text(encoding="utf-8"), encoding="utf-8")
    frame.to_csv(art_csv, index=False)
    return {
        "summary": str(summary),
        "csv": str(csv),
        "artifact_json": str(art_json),
        "artifact_csv": str(art_csv),
    }
