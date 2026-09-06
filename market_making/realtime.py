"""Event-driven single-wallet MM on live BBO updates (not candles).

Decision clock = each BookSnapshot from the order book.
Holding time = wall-clock seconds between entry fill and flat inventory.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from market_making.config import MarketMakingConfig
from market_making.engine import Quote, QuoteEngine
from market_making.fees import fee_on_notional
from market_making.feed import BookSnapshot, poll_book_events
from market_making.inventory import InventoryState

logger = logging.getLogger(__name__)


@dataclass
class OpenTrade:
    symbol: str
    entry_ts_ms: int
    entry_mid: float


@dataclass
class RealtimeResult:
    n_events: int
    n_entries: int
    n_round_trips: int
    n_buys: int
    n_sells: int
    n_skipped_busy: int
    n_multi_signal_choices: int
    fees_paid: float
    start_equity: float
    final_equity: float
    total_return_pct: float
    net_positive: bool
    hold_seconds_mean: float | None
    hold_seconds_median: float | None
    hold_seconds_p25: float | None
    hold_seconds_p75: float | None
    hold_seconds_min: float | None
    hold_seconds_max: float | None
    n_holds: int
    symbols: list[str]
    poll_interval_sec: float
    interpretation_allowed: bool
    notes: str
    holds_sec: list[float] = field(default_factory=list)


def _touch_fill_prob(
    *,
    mid: float,
    quote_px: float,
    half_bps: float,
    book_bid: float,
    book_ask: float,
    side: str,
    base_p: float,
) -> float:
    """Maker fill intensity from distance to live BBO."""
    half = max(half_bps / 1e4, 1e-8)
    if side == "buy":
        gap = max(0.0, (book_bid - quote_px) / mid)
    else:
        gap = max(0.0, (quote_px - book_ask) / mid)
    return float(np.clip(base_p * np.exp(-gap / half), 0.0, 1.0))


def _advantage(q: Quote) -> float:
    if not q.quoting:
        return float("-inf")
    return float(q.net_edge_bps_if_matched)


def run_realtime_paper(
    cfg: MarketMakingConfig,
    events: Iterable[BookSnapshot],
    *,
    seed: int = 42,
    initial_equity: float | None = None,
    mid_history_len: int = 120,
    poll_interval_sec: float | None = None,
) -> tuple[RealtimeResult, list[dict[str, Any]]]:
    """Paper MM driven by book snapshots (anlık BBO), not candle closes."""
    equity0 = float(initial_equity or cfg.portfolio_initial_equity)
    pos_frac = float(np.clip(cfg.portfolio_position_frac, 0.05, 1.0))
    flat_eps = float(cfg.portfolio_flat_eps)
    sc = cfg.sim
    rng = np.random.default_rng(seed + 101)
    poll = float(
        poll_interval_sec
        if poll_interval_sec is not None
        else getattr(cfg, "realtime_poll_sec", 1.0)
    )

    cash = equity0
    active: str | None = None
    inv = InventoryState(
        base=0.0,
        quote=0.0,
        target=0.0,
        max_abs=1.0,
        halt_frac=cfg.inventory_halt_frac,
    )
    eng = QuoteEngine(cfg)
    open_trade: OpenTrade | None = None

    hist: dict[str, deque[float]] = defaultdict(
        lambda: deque(maxlen=mid_history_len)
    )
    latest: dict[str, BookSnapshot] = {}

    n_events = 0
    n_entries = 0
    n_round_trips = 0
    n_buys = 0
    n_sells = 0
    n_skipped_busy = 0
    n_multi = 0
    fees_total = 0.0
    holds: list[float] = []
    rows: list[dict[str, Any]] = []

    def arm(sym: str, mid: float) -> None:
        nonlocal inv, eng
        max_inv = max((cash * pos_frac) / mid, 1e-12)
        inv = InventoryState(
            base=0.0,
            quote=0.0,
            target=0.0,
            max_abs=max_inv,
            halt_frac=cfg.inventory_halt_frac,
        )
        eng = QuoteEngine(cfg.model_copy(update={"max_inventory": max_inv}))

    def equity_now() -> float:
        if active is None or abs(inv.base) <= flat_eps:
            return float(cash)
        return float(cash + inv.base * latest[active].mid)

    for snap in events:
        if not snap.valid:
            continue
        n_events += 1
        latest[snap.symbol] = snap
        hist[snap.symbol].append(snap.mid)
        mid_hist = np.asarray(hist[snap.symbol], dtype=float)

        action = "idle"
        chosen: str | None = None
        n_cand = 0
        buy_fill = False
        sell_fill = False

        if active is not None:
            # Opportunity cost: other symbols that would have entered.
            for sym, s in latest.items():
                if sym == active or len(hist[sym]) < 2:
                    continue
                max_inv = max((cash * pos_frac) / s.mid, 1e-12)
                probe = QuoteEngine(cfg.model_copy(update={"max_inventory": max_inv}))
                flat_inv = InventoryState(
                    base=0.0,
                    quote=0.0,
                    target=0.0,
                    max_abs=max_inv,
                    halt_frac=cfg.inventory_halt_frac,
                )
                q_s = probe.quote(
                    s.mid, flat_inv, mid_history=np.asarray(hist[sym], dtype=float)
                )
                if not q_s.quoting or q_s.net_edge_bps_if_matched <= 0:
                    continue
                p = _touch_fill_prob(
                    mid=s.mid,
                    quote_px=q_s.bid,
                    half_bps=q_s.half_spread_bps,
                    book_bid=s.bid,
                    book_ask=s.ask,
                    side="buy",
                    base_p=sc.fill_prob,
                )
                if rng.random() < p:
                    n_skipped_busy += 1

            if snap.symbol != active:
                action = "busy_other"
            else:
                q = eng.quote(snap.mid, inv, mid_history=mid_hist)
                action = "manage"
                chosen = active
                if q.quoting and q.bid_size > 0:
                    p_bid = _touch_fill_prob(
                        mid=snap.mid,
                        quote_px=q.bid,
                        half_bps=q.half_spread_bps,
                        book_bid=snap.bid,
                        book_ask=snap.ask,
                        side="buy",
                        base_p=sc.fill_prob,
                    )
                    if rng.random() < p_bid:
                        px = q.bid * (1.0 + sc.adverse_frac * (q.half_spread_bps / 1e4))
                        fee = fee_on_notional(px * q.bid_size, cfg)
                        cost = px * q.bid_size + fee
                        if cost <= cash + 1e-9:
                            cash -= cost
                            inv.apply_buy(
                                price=px, size=q.bid_size, fee=0.0, bar=n_events
                            )
                            fees_total += fee
                            n_buys += 1
                            buy_fill = True
                if q.quoting and q.ask_size > 0:
                    sell_sz = min(q.ask_size, max(inv.base, 0.0))
                    if sell_sz > flat_eps:
                        p_ask = _touch_fill_prob(
                            mid=snap.mid,
                            quote_px=q.ask,
                            half_bps=q.half_spread_bps,
                            book_bid=snap.bid,
                            book_ask=snap.ask,
                            side="sell",
                            base_p=sc.fill_prob,
                        )
                        if rng.random() < p_ask:
                            px = q.ask * (
                                1.0 - sc.adverse_frac * (q.half_spread_bps / 1e4)
                            )
                            fee = fee_on_notional(px * sell_sz, cfg)
                            cash += px * sell_sz - fee
                            inv.apply_sell(
                                price=px, size=sell_sz, fee=0.0, bar=n_events
                            )
                            fees_total += fee
                            n_sells += 1
                            sell_fill = True

                if abs(inv.base) <= flat_eps:
                    if abs(inv.base) > 0:
                        cash += inv.base * snap.mid
                        inv.base = 0.0
                    if open_trade is not None:
                        holds.append(
                            max(0.0, (snap.ts_ms - open_trade.entry_ts_ms) / 1000.0)
                        )
                        open_trade = None
                    n_round_trips += 1
                    chosen = active
                    active = None
                    action = "close"
        else:
            cands: list[tuple[str, float, Quote, BookSnapshot]] = []
            for sym, s in latest.items():
                if len(hist[sym]) < 2:
                    continue
                max_inv = max((cash * pos_frac) / s.mid, 1e-12)
                local_eng = QuoteEngine(
                    cfg.model_copy(update={"max_inventory": max_inv})
                )
                flat_inv = InventoryState(
                    base=0.0,
                    quote=0.0,
                    target=0.0,
                    max_abs=max_inv,
                    halt_frac=cfg.inventory_halt_frac,
                )
                q = local_eng.quote(
                    s.mid, flat_inv, mid_history=np.asarray(hist[sym], dtype=float)
                )
                if not q.quoting or q.net_edge_bps_if_matched <= 0 or q.bid_size <= 0:
                    continue
                p = _touch_fill_prob(
                    mid=s.mid,
                    quote_px=q.bid,
                    half_bps=q.half_spread_bps,
                    book_bid=s.bid,
                    book_ask=s.ask,
                    side="buy",
                    base_p=sc.fill_prob,
                )
                if rng.random() < p:
                    cands.append((sym, _advantage(q), q, s))
            n_cand = len(cands)
            if n_cand > 1:
                n_multi += 1
            if cands:
                cands.sort(key=lambda t: t[1], reverse=True)
                sym, _score, _q0, s = cands[0]
                arm(sym, s.mid)
                active = sym
                chosen = sym
                n_entries += 1
                action = "enter"
                q = eng.quote(
                    s.mid, inv, mid_history=np.asarray(hist[sym], dtype=float)
                )
                if q.quoting and q.bid_size > 0:
                    px = q.bid * (1.0 + sc.adverse_frac * (q.half_spread_bps / 1e4))
                    fee = fee_on_notional(px * q.bid_size, cfg)
                    cost = px * q.bid_size + fee
                    if cost <= cash + 1e-9:
                        cash -= cost
                        inv.apply_buy(
                            price=px, size=q.bid_size, fee=0.0, bar=n_events
                        )
                        fees_total += fee
                        n_buys += 1
                        buy_fill = True
                        open_trade = OpenTrade(
                            symbol=sym, entry_ts_ms=s.ts_ms, entry_mid=s.mid
                        )
            else:
                action = "flat_wait"

        rows.append(
            {
                "event": n_events,
                "ts_ms": snap.ts_ms,
                "event_symbol": snap.symbol,
                "bid": snap.bid,
                "ask": snap.ask,
                "mid": snap.mid,
                "spread_bps": snap.spread_bps,
                "active_symbol": active,
                "action": action,
                "chosen_symbol": chosen,
                "n_candidates": n_cand,
                "cash": cash,
                "inventory": inv.base,
                "equity": equity_now(),
                "buy_fill": buy_fill,
                "sell_fill": sell_fill,
                "fees_cum": fees_total,
            }
        )

    if active is not None and abs(inv.base) > flat_eps and active in latest:
        cash += inv.base * latest[active].mid
        if open_trade is not None:
            holds.append(
                max(0.0, (latest[active].ts_ms - open_trade.entry_ts_ms) / 1000.0)
            )
        inv.base = 0.0

    final_eq = float(cash)
    ret = 100.0 * (final_eq - equity0) / equity0 if equity0 else 0.0
    arr = np.asarray(holds, dtype=float)
    result = RealtimeResult(
        n_events=n_events,
        n_entries=n_entries,
        n_round_trips=n_round_trips,
        n_buys=n_buys,
        n_sells=n_sells,
        n_skipped_busy=n_skipped_busy,
        n_multi_signal_choices=n_multi,
        fees_paid=float(fees_total),
        start_equity=float(equity0),
        final_equity=final_eq,
        total_return_pct=float(ret),
        net_positive=bool(final_eq > equity0 and n_entries > 0),
        hold_seconds_mean=float(arr.mean()) if len(arr) else None,
        hold_seconds_median=float(np.median(arr)) if len(arr) else None,
        hold_seconds_p25=float(np.percentile(arr, 25)) if len(arr) else None,
        hold_seconds_p75=float(np.percentile(arr, 75)) if len(arr) else None,
        hold_seconds_min=float(arr.min()) if len(arr) else None,
        hold_seconds_max=float(arr.max()) if len(arr) else None,
        n_holds=int(len(arr)),
        symbols=list(cfg.symbols),
        poll_interval_sec=poll,
        interpretation_allowed=False,
        notes=(
            "Event-driven MM on BBO snapshots (NOT candles). "
            f"Fee {cfg.commission_pct}%/side. Hold time = wall-clock seconds. "
            "Not a live-edge claim."
        ),
        holds_sec=holds,
    )
    return result, rows


def run_live_paper_session(
    cfg: MarketMakingConfig,
    *,
    duration_sec: float = 120.0,
    poll_interval_sec: float | None = None,
    seed: int = 42,
    initial_equity: float | None = None,
    exchange_id: str | None = None,
) -> tuple[RealtimeResult, list[dict[str, Any]]]:
    """Poll live spot books and run paper MM for ``duration_sec``."""
    poll = float(
        poll_interval_sec
        if poll_interval_sec is not None
        else getattr(cfg, "realtime_poll_sec", 1.0)
    )
    max_cycles = max(1, int(duration_sec / max(poll, 0.2)))
    events = poll_book_events(
        list(cfg.symbols),
        exchange_id=exchange_id or cfg.exchange_id,
        poll_interval_sec=poll,
        max_cycles=max_cycles,
    )
    return run_realtime_paper(
        cfg,
        events,
        seed=seed,
        initial_equity=initial_equity,
        poll_interval_sec=poll,
    )


def save_realtime_artifacts(
    result: RealtimeResult,
    rows: list[dict[str, Any]],
    *,
    root: str | Path | None = None,
    exchange_id: str = "okx",
    basename: str = "realtime_paper_latest",
) -> dict[str, str]:
    out = Path(root or "data") / exchange_id / "market_making"
    out.mkdir(parents=True, exist_ok=True)
    summary = out / f"{basename}.json"
    csv = out / f"{basename}.csv"
    summary.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(csv, index=False)
    return {"summary": str(summary), "csv": str(csv)}
