"""$200 dual-slot live-BBO paper bot (NOT live orders).

Evidence step (live edge ladder)
--------------------------------
1. Candle paper          → regime / param sanity (done; not live-edge)
2. Live BBO paper fills  → THIS MODULE (real books, simulated maker fills)
3. Tiny live maker orders → real edge claim only after sustained step-2 + step-3

Capital rule
------------
Start $200. Fixed $100 windows; remnant ≥ $100 opens another window
→ two $100 slots at start. No monthly DCA.

interpretation_allowed=false until real fills + fees + inventory sync
survive a pre-declared observation window.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from market_making.config import load_mm_config
from market_making.engine import Quote, QuoteEngine
from market_making.fees import fee_on_notional
from market_making.feed import BookSnapshot, poll_book_events
from market_making.inventory import InventoryState
from market_making.realtime import _touch_fill_prob
from market_making.run_retail_5m_250slots import slot_plan, slot_size

START_EQUITY = 200.0
SLOT_MIN = 100.0
SLOT_MAX = 100.0
REMNANT_MIN = 100.0


@dataclass
class SlotState:
    symbol: str
    inv: InventoryState
    eng: QuoteEngine
    notional_budget: float
    entry_ts_ms: int | None = None


@dataclass
class EvidenceResult:
    n_events: int
    n_entries: int
    n_round_trips: int
    n_buys: int
    n_sells: int
    n_busy: int
    fees_paid: float
    start_equity: float
    final_equity: float
    total_return_pct: float
    peak_slots: int
    mean_open_slots: float
    hold_seconds_mean: float | None
    hold_seconds_median: float | None
    n_holds: int
    duration_sec: float
    poll_sec: float
    n_symbols: int
    fill_prob: float
    adverse_frac: float
    interpretation_allowed: bool
    evidence_tier: str
    notes: str
    holds_sec: list[float] = field(default_factory=list)


def _score(q: Quote) -> float:
    if not q.quoting:
        return float("-inf")
    return float(q.net_edge_bps_if_matched)


def run_dual_slot_live_paper(
    cfg,
    events,
    *,
    seed: int = 42,
    initial_equity: float = START_EQUITY,
    poll_sec: float = 2.0,
    duration_sec: float = 0.0,
    mid_history_len: int = 240,
) -> tuple[EvidenceResult, list[dict[str, Any]]]:
    sc = cfg.sim
    rng = np.random.default_rng(seed + 202)
    flat_eps = float(cfg.portfolio_flat_eps)
    cash = float(initial_equity)
    slots: dict[str, SlotState] = {}
    hist: dict[str, list[float]] = {s: [] for s in cfg.symbols}
    latest: dict[str, BookSnapshot] = {}

    n_events = n_entries = n_rt = n_buys = n_sells = n_busy = 0
    fees_total = 0.0
    peak_slots = 0
    open_hist: list[int] = []
    holds: list[float] = []
    rows: list[dict[str, Any]] = []
    t0 = time.time()

    def mark(mids: dict[str, float]) -> float:
        eq = cash
        for sym, slot in slots.items():
            eq += slot.inv.base * mids[sym]
        return float(eq)

    def arm(sym: str, mid: float, budget: float, ts_ms: int) -> SlotState:
        max_inv = max(budget / mid, 1e-12)
        inv = InventoryState(
            base=0.0,
            quote=0.0,
            target=0.0,
            max_abs=max_inv,
            halt_frac=cfg.inventory_halt_frac,
        )
        eng = QuoteEngine(cfg.model_copy(update={"max_inventory": max_inv}))
        return SlotState(
            symbol=sym, inv=inv, eng=eng, notional_budget=budget, entry_ts_ms=ts_ms
        )

    for snap in events:
        if not snap.valid:
            continue
        n_events += 1
        latest[snap.symbol] = snap
        h = hist.setdefault(snap.symbol, [])
        h.append(snap.mid)
        if len(h) > mid_history_len:
            del h[: len(h) - mid_history_len]

        mids = {s: float(latest[s].mid) for s in latest}
        closed: list[str] = []
        opened: list[str] = []

        # Manage open slots on their own book updates (and lightly on others).
        for sym in list(slots.keys()):
            if sym not in latest:
                continue
            book = latest[sym]
            slot = slots[sym]
            mid = book.mid
            q = slot.eng.quote(
                mid, slot.inv, mid_history=np.asarray(hist[sym], dtype=float)
            )
            if q.quoting:
                if q.bid_size > 0:
                    p = _touch_fill_prob(
                        mid=mid,
                        quote_px=q.bid,
                        half_bps=q.half_spread_bps,
                        book_bid=book.bid,
                        book_ask=book.ask,
                        side="buy",
                        base_p=sc.fill_prob,
                    )
                    if rng.random() < p:
                        px = q.bid * (
                            1.0 + sc.adverse_frac * (q.half_spread_bps / 1e4)
                        )
                        fee = fee_on_notional(px * q.bid_size, cfg)
                        cost = px * q.bid_size + fee
                        if cost <= cash + 1e-9 and (
                            slot.inv.base * mid + cost
                            <= slot.notional_budget * 1.05 + 1e-9
                        ):
                            cash -= cost
                            slot.inv.apply_buy(
                                price=px, size=q.bid_size, fee=0.0, bar=n_events
                            )
                            fees_total += fee
                            n_buys += 1
                if q.ask_size > 0 and slot.inv.base > flat_eps:
                    p = _touch_fill_prob(
                        mid=mid,
                        quote_px=q.ask,
                        half_bps=q.half_spread_bps,
                        book_bid=book.bid,
                        book_ask=book.ask,
                        side="sell",
                        base_p=sc.fill_prob,
                    )
                    if rng.random() < p:
                        sell_sz = min(q.ask_size, slot.inv.base)
                        px = q.ask * (
                            1.0 - sc.adverse_frac * (q.half_spread_bps / 1e4)
                        )
                        fee = fee_on_notional(px * sell_sz, cfg)
                        cash += px * sell_sz - fee
                        slot.inv.apply_sell(
                            price=px, size=sell_sz, fee=0.0, bar=n_events
                        )
                        fees_total += fee
                        n_sells += 1
            if abs(slot.inv.base) <= flat_eps:
                if abs(slot.inv.base) > 0:
                    cash += slot.inv.base * mid
                    slot.inv.base = 0.0
                if slot.entry_ts_ms is not None:
                    holds.append(max(0.0, (book.ts_ms - slot.entry_ts_ms) / 1000.0))
                n_rt += 1
                closed.append(sym)
                del slots[sym]

        eq_now = mark(mids)
        plan = slot_plan(
            eq_now,
            start_equity=initial_equity,
            slot_min=SLOT_MIN,
            slot_max=SLOT_MAX,
            remnant_min=REMNANT_MIN,
        )
        max_slots = max(len(plan), 1)
        peak_slots = max(peak_slots, max_slots)
        free = max_slots - len(slots)

        if free <= 0:
            # Count a busy skip if another symbol looks quotable.
            if snap.symbol not in slots and len(hist.get(snap.symbol, [])) >= 5:
                bud = plan[-1] if plan else SLOT_MIN
                max_inv = max(bud / snap.mid, 1e-12)
                probe = QuoteEngine(cfg.model_copy(update={"max_inventory": max_inv}))
                flat = InventoryState(
                    base=0.0,
                    quote=0.0,
                    target=0.0,
                    max_abs=max_inv,
                    halt_frac=cfg.inventory_halt_frac,
                )
                q = probe.quote(
                    snap.mid,
                    flat,
                    mid_history=np.asarray(hist[snap.symbol], dtype=float),
                )
                if q.quoting and q.net_edge_bps_if_matched > 0:
                    n_busy += 1
        elif plan:
            used = sorted((sl.notional_budget for sl in slots.values()), reverse=True)
            remaining = list(plan)
            for b in used:
                for j, pbud in enumerate(remaining):
                    if abs(pbud - b) < 1e-6:
                        remaining.pop(j)
                        break
            if remaining:
                probe_budget = remaining[0]
                cands: list[tuple[str, float, Quote, BookSnapshot]] = []
                for s, book in latest.items():
                    if s in slots or len(hist.get(s, [])) < 5:
                        continue
                    max_inv = max(probe_budget / book.mid, 1e-12)
                    local = QuoteEngine(
                        cfg.model_copy(update={"max_inventory": max_inv})
                    )
                    flat = InventoryState(
                        base=0.0,
                        quote=0.0,
                        target=0.0,
                        max_abs=max_inv,
                        halt_frac=cfg.inventory_halt_frac,
                    )
                    q = local.quote(
                        book.mid,
                        flat,
                        mid_history=np.asarray(hist[s], dtype=float),
                    )
                    if (
                        not q.quoting
                        or q.net_edge_bps_if_matched <= 0
                        or q.bid_size <= 0
                    ):
                        continue
                    p = _touch_fill_prob(
                        mid=book.mid,
                        quote_px=q.bid,
                        half_bps=q.half_spread_bps,
                        book_bid=book.bid,
                        book_ask=book.ask,
                        side="buy",
                        base_p=sc.fill_prob,
                    )
                    if rng.random() < p:
                        cands.append((s, _score(q), q, book))
                cands.sort(key=lambda t: t[1], reverse=True)
                for sym, _sc0, q0, book in cands:
                    if free <= 0 or not remaining:
                        break
                    budget = remaining.pop(0)
                    max_inv = max(budget / book.mid, 1e-12)
                    local = QuoteEngine(
                        cfg.model_copy(update={"max_inventory": max_inv})
                    )
                    flat = InventoryState(
                        base=0.0,
                        quote=0.0,
                        target=0.0,
                        max_abs=max_inv,
                        halt_frac=cfg.inventory_halt_frac,
                    )
                    q1 = local.quote(
                        book.mid,
                        flat,
                        mid_history=np.asarray(hist[sym], dtype=float),
                    )
                    if not q1.quoting or q1.bid_size <= 0:
                        remaining.insert(0, budget)
                        continue
                    px = q1.bid * (
                        1.0 + sc.adverse_frac * (q1.half_spread_bps / 1e4)
                    )
                    fee = fee_on_notional(px * q1.bid_size, cfg)
                    cost = px * q1.bid_size + fee
                    if cost > cash + 1e-9:
                        remaining.insert(0, budget)
                        continue
                    slot = arm(sym, book.mid, budget, book.ts_ms)
                    cash -= cost
                    slot.inv.apply_buy(
                        price=px, size=q1.bid_size, fee=0.0, bar=n_events
                    )
                    fees_total += fee
                    slots[sym] = slot
                    n_entries += 1
                    n_buys += 1
                    opened.append(sym)
                    free -= 1

        open_hist.append(len(slots))
        if n_events % 50 == 0 or opened or closed:
            rows.append(
                {
                    "event": n_events,
                    "ts_ms": snap.ts_ms,
                    "equity": mark(mids),
                    "cash": cash,
                    "n_open": len(slots),
                    "open_symbols": ",".join(sorted(slots.keys())),
                    "opened": ",".join(opened),
                    "closed": ",".join(closed),
                    "fees_cum": fees_total,
                }
            )

        if duration_sec > 0 and (time.time() - t0) >= duration_sec:
            break

    # Flatten residual inventory at last mids.
    for sym, slot in list(slots.items()):
        mid = latest[sym].mid if sym in latest else 0.0
        cash += slot.inv.base * mid
        if slot.entry_ts_ms is not None and sym in latest:
            holds.append(
                max(0.0, (latest[sym].ts_ms - slot.entry_ts_ms) / 1000.0)
            )
        n_rt += 1
        del slots[sym]

    final_eq = float(cash)
    ret = 100.0 * (final_eq - initial_equity) / initial_equity if initial_equity else 0.0
    arr = np.asarray(holds, dtype=float)
    result = EvidenceResult(
        n_events=n_events,
        n_entries=n_entries,
        n_round_trips=n_rt,
        n_buys=n_buys,
        n_sells=n_sells,
        n_busy=n_busy,
        fees_paid=float(fees_total),
        start_equity=float(initial_equity),
        final_equity=final_eq,
        total_return_pct=float(ret),
        peak_slots=peak_slots,
        mean_open_slots=float(np.mean(open_hist)) if open_hist else 0.0,
        hold_seconds_mean=float(arr.mean()) if len(arr) else None,
        hold_seconds_median=float(np.median(arr)) if len(arr) else None,
        n_holds=int(len(arr)),
        duration_sec=float(time.time() - t0),
        poll_sec=float(poll_sec),
        n_symbols=len(cfg.symbols),
        fill_prob=float(sc.fill_prob),
        adverse_frac=float(sc.adverse_frac),
        interpretation_allowed=False,
        evidence_tier="live_bbo_paper_fills",
        notes=(
            "Tier-2 evidence: live OKX BBO, paper maker fills with adverse selection. "
            "NOT live orders / NOT a live-edge claim. "
            f"$200 start, fixed $100 windows (2 at start). Fee {cfg.commission_pct}%/side."
        ),
        holds_sec=holds,
    )
    return result, rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="$200 dual-slot live-BBO paper evidence bot")
    p.add_argument("--config", default="config/market_making_5m.yaml")
    p.add_argument("--equity", type=float, default=START_EQUITY)
    p.add_argument("--duration-sec", type=float, default=900.0)
    p.add_argument("--poll-sec", type=float, default=2.0)
    p.add_argument("--max-symbols", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-basename", default="live_edge_tier2_200_dualslot")
    args = p.parse_args(argv)

    cfg = load_mm_config(args.config)
    # Faster REST poll: focus on liquid names for evidence session.
    symbols = list(cfg.symbols)[: max(2, int(args.max_symbols))]
    cfg = cfg.model_copy(
        update={
            "symbols": symbols,
            "portfolio_initial_equity": float(args.equity),
        }
    )

    print(
        f"Tier-2 live-BBO paper: ${args.equity:.0f} start, "
        f"2×$100 windows, {len(symbols)} symbols, "
        f"{args.duration_sec:.0f}s @ poll {args.poll_sec}s "
        f"(fill_prob={cfg.sim.fill_prob}, adverse={cfg.sim.adverse_frac})",
        flush=True,
    )
    print("interpretation_allowed=false — not a live-edge claim", flush=True)

    max_cycles = max(1, int(float(args.duration_sec) / max(float(args.poll_sec), 0.2)))
    events = poll_book_events(
        symbols,
        exchange_id=cfg.exchange_id,
        poll_interval_sec=float(args.poll_sec),
        max_cycles=max_cycles,
    )
    result, rows = run_dual_slot_live_paper(
        cfg,
        events,
        seed=int(args.seed),
        initial_equity=float(args.equity),
        poll_sec=float(args.poll_sec),
        duration_sec=float(args.duration_sec),
    )

    out = {
        "interpretation_allowed": False,
        "evidence_tier": result.evidence_tier,
        "design": {
            "start_equity": float(args.equity),
            "slot": 100.0,
            "windows_at_start": slot_plan(
                float(args.equity),
                start_equity=float(args.equity),
                slot_min=100.0,
                slot_max=100.0,
                remnant_min=100.0,
            ),
            "config": args.config,
            "symbols": symbols,
        },
        "result": asdict(result),
    }
    Path(f"config/{args.out_basename}_results.json").write_text(
        json.dumps(out, indent=2, default=str)
    )
    art = Path("/opt/cursor/artifacts")
    try:
        art.mkdir(parents=True, exist_ok=True)
        (art / f"{args.out_basename}_results.json").write_text(
            json.dumps(out, indent=2, default=str)
        )
        pd.DataFrame(rows).to_csv(art / f"{args.out_basename}_events.csv", index=False)
    except OSError as e:
        print(f"artifact write skipped: {e}", flush=True)

    data = Path(cfg.storage_root) / cfg.exchange_id / "market_making"
    data.mkdir(parents=True, exist_ok=True)
    (data / f"{args.out_basename}.json").write_text(
        json.dumps(out, indent=2, default=str)
    )
    pd.DataFrame(rows).to_csv(data / f"{args.out_basename}.csv", index=False)

    print(
        f"  → {result.start_equity:.0f} → {result.final_equity:.2f} "
        f"({result.total_return_pct:+.2f}%) entries={result.n_entries} "
        f"rt={result.n_round_trips} busy={result.n_busy} "
        f"hold_med={result.hold_seconds_median} "
        f"fees={result.fees_paid:.4f} events={result.n_events}",
        flush=True,
    )
    print(f"Wrote config/{args.out_basename}_results.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
