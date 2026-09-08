"""$200 dual-slot live paper bot — 5m candle clock (decision @ 4:59).

Evidence step (live edge ladder)
--------------------------------
1. Candle paper (hist 5m close) → regime / param sanity
2. Live candle-clock paper fills → THIS MODULE
   - Poll BBO, but **trade only once per 5m bar**
   - Decision price = mid at second 299 of the bar (4m59s)
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
from typing import Any, Iterator

import numpy as np
import pandas as pd

from market_making.config import load_mm_config
from market_making.engine import Quote, QuoteEngine
from market_making.fees import fee_on_notional
from market_making.feed import BookSnapshot, connect_spot_exchange, snapshot_from_order_book
from market_making.inventory import InventoryState
from market_making.realtime import _touch_fill_prob
from market_making.run_retail_5m_250slots import slot_plan

START_EQUITY = 200.0
SLOT_MIN = 100.0
SLOT_MAX = 100.0
REMNANT_MIN = 100.0
BAR_SEC = 300
DECISION_SEC_IN_BAR = 299  # 4m59s within 5m bar; 1m uses 59


def configure_clock(bar_sec: int, decision_sec: int) -> None:
    """Set candle decision clock (e.g. 300/299 for 5m, 60/59 for 1m)."""
    global BAR_SEC, DECISION_SEC_IN_BAR
    BAR_SEC = int(bar_sec)
    DECISION_SEC_IN_BAR = int(decision_sec)


@dataclass
class SlotState:
    symbol: str
    inv: InventoryState
    eng: QuoteEngine
    notional_budget: float
    entry_bar_id: int | None = None


@dataclass
class EvidenceResult:
    n_polls: int
    n_decision_bars: int
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
    hold_bars_mean: float | None
    hold_bars_median: float | None
    n_holds: int
    duration_sec: float
    poll_sec: float
    n_symbols: int
    fill_prob: float
    adverse_frac: float
    decision_sec_in_bar: int
    interpretation_allowed: bool
    evidence_tier: str
    notes: str
    holds_bars: list[float] = field(default_factory=list)


def _score(q: Quote) -> float:
    if not q.quoting:
        return float("-inf")
    return float(q.net_edge_bps_if_matched)


def _bar_id(ts_ms: int) -> int:
    return int(ts_ms // (BAR_SEC * 1000))


def _sec_in_bar(ts_ms: int) -> int:
    return int((ts_ms // 1000) % BAR_SEC)


def poll_books_once(
    exchange, symbols: list[str], *, exchange_id: str
) -> dict[str, BookSnapshot]:
    out: dict[str, BookSnapshot] = {}
    now_ms = int(time.time() * 1000)
    for sym in symbols:
        try:
            book = exchange.fetch_order_book(sym, limit=5)
        except Exception:
            continue
        snap = snapshot_from_order_book(
            sym, book, exchange_id=exchange_id, ts_ms=now_ms
        )
        if snap is not None and snap.valid:
            out[sym] = snap
    return out


def iter_candle_decision_books(
    symbols: list[str],
    *,
    exchange_id: str = "okx",
    poll_interval_sec: float = 1.0,
    duration_sec: float = 900.0,
    decision_sec: int = DECISION_SEC_IN_BAR,
) -> Iterator[tuple[int, dict[str, BookSnapshot]]]:
    """Yield (bar_id, books) once per 5m bar when wall clock hits ≥ decision_sec."""
    connected = connect_spot_exchange(exchange_id)
    exchange = connected.exchange
    eid = connected.exchange_id
    t0 = time.time()
    last_decided: int | None = None
    n_polls = 0
    print(
        f"Candle clock live: bar={BAR_SEC}s decision_sec>={decision_sec} "
        f"poll={poll_interval_sec}s, duration={duration_sec:.0f}s",
        flush=True,
    )
    while True:
        if duration_sec > 0 and (time.time() - t0) >= duration_sec:
            break
        now_ms = int(time.time() * 1000)
        bid = _bar_id(now_ms)
        sec = _sec_in_bar(now_ms)
        n_polls += 1
        if sec >= decision_sec and bid != last_decided:
            books = poll_books_once(exchange, symbols, exchange_id=eid)
            if books:
                last_decided = bid
                print(
                    f"  decision bar_id={bid} sec_in_bar={sec} "
                    f"n_books={len(books)} (polls={n_polls})",
                    flush=True,
                )
                yield bid, books
            else:
                print(
                    f"  skip empty books at bar_id={bid} sec={sec}",
                    flush=True,
                )
        # Sleep until near next decision or next poll.
        now_ms = int(time.time() * 1000)
        sec = _sec_in_bar(now_ms)
        if sec < decision_sec:
            wait = min(poll_interval_sec, max(0.2, decision_sec - sec - 0.05))
        else:
            # Past decision window: nap until next bar starts.
            wait = min(poll_interval_sec, max(0.2, BAR_SEC - sec))
        if duration_sec > 0:
            rem = duration_sec - (time.time() - t0)
            if rem <= 0:
                break
            wait = min(wait, rem)
        time.sleep(wait)


def run_dual_slot_candle_paper(
    cfg,
    decisions: Iterator[tuple[int, dict[str, BookSnapshot]]],
    *,
    seed: int = 42,
    initial_equity: float = START_EQUITY,
    poll_sec: float = 1.0,
    mid_history_len: int = 240,
    seed_hist: dict[str, list[float]] | None = None,
) -> tuple[EvidenceResult, list[dict[str, Any]]]:
    sc = cfg.sim
    rng = np.random.default_rng(seed + 202)
    flat_eps = float(cfg.portfolio_flat_eps)
    cash = float(initial_equity)
    slots: dict[str, SlotState] = {}
    hist: dict[str, list[float]] = {s: [] for s in cfg.symbols}
    if seed_hist:
        for s, vals in seed_hist.items():
            hist[s] = list(vals)[-mid_history_len:]
    latest: dict[str, BookSnapshot] = {}

    n_polls = 0
    n_bars = n_entries = n_rt = n_buys = n_sells = n_busy = 0
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

    def arm(sym: str, mid: float, budget: float, bar_id: int) -> SlotState:
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
            symbol=sym, inv=inv, eng=eng, notional_budget=budget, entry_bar_id=bar_id
        )

    for bar_id, books in decisions:
        n_bars += 1
        n_polls += 1
        for sym, snap in books.items():
            latest[sym] = snap
            h = hist.setdefault(sym, [])
            h.append(snap.mid)
            if len(h) > mid_history_len:
                del h[: len(h) - mid_history_len]

        mids = {s: float(latest[s].mid) for s in latest}
        closed: list[str] = []
        opened: list[str] = []

        # Manage open slots once per candle decision.
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
                                price=px, size=q.bid_size, fee=0.0, bar=n_bars
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
                            price=px, size=sell_sz, fee=0.0, bar=n_bars
                        )
                        fees_total += fee
                        n_sells += 1
            if abs(slot.inv.base) <= flat_eps:
                if abs(slot.inv.base) > 0:
                    cash += slot.inv.base * mid
                    slot.inv.base = 0.0
                if slot.entry_bar_id is not None:
                    holds.append(float(max(0, bar_id - slot.entry_bar_id)))
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
            for s, book in latest.items():
                if s in slots or len(hist.get(s, [])) < 5:
                    continue
                bud = plan[-1] if plan else SLOT_MIN
                max_inv = max(bud / book.mid, 1e-12)
                probe = QuoteEngine(cfg.model_copy(update={"max_inventory": max_inv}))
                flat = InventoryState(
                    base=0.0,
                    quote=0.0,
                    target=0.0,
                    max_abs=max_inv,
                    halt_frac=cfg.inventory_halt_frac,
                )
                q = probe.quote(
                    book.mid,
                    flat,
                    mid_history=np.asarray(hist[s], dtype=float),
                )
                if q.quoting and q.net_edge_bps_if_matched > 0:
                    n_busy += 1
                    break
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
                    slot = arm(sym, book.mid, budget, bar_id)
                    cash -= cost
                    slot.inv.apply_buy(
                        price=px, size=q1.bid_size, fee=0.0, bar=n_bars
                    )
                    fees_total += fee
                    slots[sym] = slot
                    n_entries += 1
                    n_buys += 1
                    opened.append(sym)
                    free -= 1

        open_hist.append(len(slots))
        rows.append(
            {
                "decision_bar": n_bars,
                "bar_id": bar_id,
                "ts_ms": next(iter(books.values())).ts_ms,
                "equity": mark(mids),
                "cash": cash,
                "n_open": len(slots),
                "open_symbols": ",".join(sorted(slots.keys())),
                "opened": ",".join(opened),
                "closed": ",".join(closed),
                "fees_cum": fees_total,
            }
        )

    # Flatten residual inventory at last mids.
    for sym, slot in list(slots.items()):
        mid = latest[sym].mid if sym in latest else 0.0
        cash += slot.inv.base * mid
        if slot.entry_bar_id is not None:
            holds.append(0.0)
        n_rt += 1
        del slots[sym]

    final_eq = float(cash)
    ret = 100.0 * (final_eq - initial_equity) / initial_equity if initial_equity else 0.0
    arr = np.asarray(holds, dtype=float)
    result = EvidenceResult(
        n_polls=n_polls,
        n_decision_bars=n_bars,
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
        hold_bars_mean=float(arr.mean()) if len(arr) else None,
        hold_bars_median=float(np.median(arr)) if len(arr) else None,
        n_holds=int(len(arr)),
        duration_sec=float(time.time() - t0),
        poll_sec=float(poll_sec),
        n_symbols=len(cfg.symbols),
        fill_prob=float(sc.fill_prob),
        adverse_frac=float(sc.adverse_frac),
        decision_sec_in_bar=DECISION_SEC_IN_BAR,
        interpretation_allowed=False,
        evidence_tier="live_candle_clock_paper_fills",
        notes=(
            "Tier-2 candle-clock: live OKX BBO sampled at second 299 of each 5m bar "
            "(4m59s). One decision per candle. Paper fills only — NOT live orders / "
            f"NOT a live-edge claim. $200 start, fixed $100 windows. "
            f"Fee {cfg.commission_pct}%/side."
        ),
        holds_bars=holds,
    )
    return result, rows


def _seed_hist_from_parquet(
    symbols: list[str],
    *,
    timeframe: str,
    n_bars: int,
    root: str | Path,
    exchange_id: str,
) -> dict[str, list[float]]:
    """Warm mid history from recent candle closes so live can trade immediately."""
    from market_making.simulator import load_okx_mids
    import pandas as pd

    end = pd.Timestamp.now(tz="UTC").ceil("min")
    start = end - pd.Timedelta(minutes=max(n_bars + 5, 30))
    out: dict[str, list[float]] = {}
    for sym in symbols:
        try:
            mids, _ = load_okx_mids(
                sym,
                timeframe=timeframe,
                start=start.isoformat(),
                end_exclusive=end.isoformat(),
                root=root,
                exchange_id=exchange_id,
            )
            out[sym] = [float(x) for x in mids[-n_bars:]]
        except Exception as exc:  # noqa: BLE001
            print(f"  warmup skip {sym}: {exc}", flush=True)
            out[sym] = []
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="$200 dual-slot live candle-clock paper"
    )
    p.add_argument("--config", default="config/market_making_5m.yaml")
    p.add_argument("--equity", type=float, default=START_EQUITY)
    p.add_argument("--duration-sec", type=float, default=1200.0)
    p.add_argument("--poll-sec", type=float, default=1.0)
    p.add_argument("--max-symbols", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bar-sec", type=int, default=300)
    p.add_argument("--decision-sec", type=int, default=299)
    p.add_argument("--warmup-bars", type=int, default=0)
    p.add_argument("--warmup-timeframe", default="")
    p.add_argument("--out-basename", default="live_candle_5m_200_dualslot")
    args = p.parse_args(argv)

    configure_clock(int(args.bar_sec), int(args.decision_sec))
    cfg = load_mm_config(args.config)
    symbols = list(cfg.symbols)[: max(2, int(args.max_symbols))]
    cfg = cfg.model_copy(
        update={
            "symbols": symbols,
            "portfolio_initial_equity": float(args.equity),
        }
    )

    print(
        f"Tier-2 live candle-clock paper: ${args.equity:.0f} start, "
        f"2×$100 windows, {len(symbols)} symbols, "
        f"{args.duration_sec:.0f}s, decision @ {DECISION_SEC_IN_BAR}s/{BAR_SEC} "
        f"(fill_prob={cfg.sim.fill_prob}, adverse={cfg.sim.adverse_frac})",
        flush=True,
    )
    print("interpretation_allowed=false — not a live-edge claim", flush=True)

    seed_hist: dict[str, list[float]] | None = None
    if int(args.warmup_bars) > 0:
        tf = args.warmup_timeframe.strip() or (
            "1m" if BAR_SEC <= 60 else "5m" if BAR_SEC <= 300 else "1h"
        )
        print(f"Warming mid history from {tf} parquet ({args.warmup_bars} bars)...", flush=True)
        seed_hist = _seed_hist_from_parquet(
            symbols,
            timeframe=tf,
            n_bars=int(args.warmup_bars),
            root=cfg.storage_root,
            exchange_id=cfg.exchange_id,
        )
        n_ok = sum(1 for v in seed_hist.values() if len(v) >= 5)
        print(f"  warmup ready symbols={n_ok}/{len(symbols)}", flush=True)

    decisions = iter_candle_decision_books(
        symbols,
        exchange_id=cfg.exchange_id,
        poll_interval_sec=float(args.poll_sec),
        duration_sec=float(args.duration_sec),
        decision_sec=DECISION_SEC_IN_BAR,
    )
    result, rows = run_dual_slot_candle_paper(
        cfg,
        decisions,
        seed=int(args.seed),
        initial_equity=float(args.equity),
        poll_sec=float(args.poll_sec),
        seed_hist=seed_hist,
    )

    label = (
        f"{DECISION_SEC_IN_BAR // 60}m{DECISION_SEC_IN_BAR % 60:02d}s"
        if BAR_SEC >= 60
        else f"{DECISION_SEC_IN_BAR}s"
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
            "decision_clock": {
                "bar_seconds": BAR_SEC,
                "decision_second_in_bar": DECISION_SEC_IN_BAR,
                "decision_label": label,
            },
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
        f"decision_bars={result.n_decision_bars} "
        f"hold_bars_med={result.hold_bars_median} "
        f"fees={result.fees_paid:.4f}",
        flush=True,
    )
    print(f"Wrote config/{args.out_basename}_results.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
