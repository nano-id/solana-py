"""Retail MM paper: 5m candles, $250 start, windows scale $100→$250.

Slot rule
---------
Base slot size scales linearly with mark equity:
  equity ≤ $250  → slot $100
  equity ≥ $1250 → slot $250
  in between     → linear
Full slots of that size are opened; if remainder ≥ $100, an extra remnant
window opens (same pattern as the $2k/$500 rule, scaled down).

Deposits
--------
Start $250. Optional monthly DCA (default off; use --monthly-deposit 250).
Extra remnant window when remainder ≥ $100. At $250 equity → two $100 windows.

Forward window: 2026-01-01 → 2026-09-07, OKX 5m.
interpretation_allowed=false.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from market_making.config import load_mm_config
from market_making.engine import Quote, QuoteEngine
from market_making.fees import fee_on_notional
from market_making.inventory import InventoryState
from market_making.portfolio import load_aligned_panel
from market_making.simulator import _fill_prob

START_EQUITY = 250.0
MONTHLY_DEPOSIT = 0.0
SLOT_MIN = 100.0
SLOT_MAX = 250.0
REMNANT_MIN = 100.0
SLOT_MAX_AT_EQUITY = 1_250.0


def slot_size(equity: float) -> float:
    eq = max(float(equity), 0.0)
    if eq <= START_EQUITY:
        return SLOT_MIN
    if eq >= SLOT_MAX_AT_EQUITY:
        return SLOT_MAX
    return SLOT_MIN + (SLOT_MAX - SLOT_MIN) * (eq - START_EQUITY) / (
        SLOT_MAX_AT_EQUITY - START_EQUITY
    )


def slot_plan(equity: float) -> list[float]:
    eq = max(float(equity), 0.0)
    base = slot_size(eq)
    if base <= 0:
        return []
    full = int(eq // base)
    rem = eq - full * base
    if full <= 0 and rem >= REMNANT_MIN - 1e-9:
        return [rem]
    if rem >= REMNANT_MIN - 1e-9:
        return [base] * full + [rem]
    return [base] * max(full, 0)


@dataclass
class SlotState:
    symbol: str
    inv: InventoryState
    eng: QuoteEngine
    notional_budget: float


@dataclass
class Result:
    n_bars: int
    n_entries: int
    n_round_trips: int
    n_buys: int
    n_sells: int
    n_skipped_busy: int
    n_multi_signal_choices: int
    peak_slots: int
    start_equity: float
    final_equity: float
    total_deposited: float
    net_profit: float
    return_on_deposited_pct: float
    total_return_vs_start_pct: float
    fees_paid: float
    realized_pnl: float
    n_active_symbols: int
    active_symbol_bars: dict[str, int]
    symbols: list[str]
    max_open_slots_mean: float
    n_deposits: int
    interpretation_allowed: bool
    notes: str
    open_slots_end: list[str] = field(default_factory=list)


def _score(q: Quote) -> float:
    if not q.quoting:
        return float("-inf")
    return float(q.net_edge_bps_if_matched)


def run_dynamic(
    cfg,
    panel: dict[str, np.ndarray],
    timestamps,
    *,
    seed: int = 42,
    initial_equity: float = START_EQUITY,
    monthly_deposit: float = MONTHLY_DEPOSIT,
) -> tuple[Result, pd.DataFrame]:
    symbols = list(panel.keys())
    n = len(next(iter(panel.values())))
    flat_eps = float(cfg.portfolio_flat_eps)
    sc = cfg.sim
    rng = np.random.default_rng(seed + 77)
    look = max(int(cfg.vol_lookback), 8)
    mid_mat = {s: np.asarray(panel[s], dtype=float) for s in symbols}

    cash = float(initial_equity)
    total_deposited = float(initial_equity)
    n_deposits = 0
    slots: dict[str, SlotState] = {}

    n_entries = n_rt = n_buys = n_sells = n_busy = n_multi = 0
    fees_total = realized_total = 0.0
    active_bars = {s: 0 for s in symbols}
    peak_slots = 0
    open_hist: list[int] = []
    rows: list[dict[str, Any]] = []

    ts_index = pd.to_datetime(pd.Index(timestamps), utc=True)
    prev_ym: str | None = None
    record_every = 12

    def mark_equity(mids: dict[str, float]) -> float:
        eq = cash
        for sym, slot in slots.items():
            eq += slot.inv.base * mids[sym]
        return float(eq)

    def arm(sym: str, mid: float, budget: float) -> SlotState:
        max_inv = max(budget / mid, 1e-12)
        inv = InventoryState(
            base=0.0,
            quote=0.0,
            target=0.0,
            max_abs=max_inv,
            halt_frac=cfg.inventory_halt_frac,
        )
        eng = QuoteEngine(cfg.model_copy(update={"max_inventory": max_inv}))
        return SlotState(symbol=sym, inv=inv, eng=eng, notional_budget=budget)

    def mid_window(sym: str, i: int) -> np.ndarray:
        lo = 0 if i + 1 <= look else i + 1 - look
        return mid_mat[sym][lo : i + 1]

    for i in range(n):
        mids = {s: float(mid_mat[s][i]) for s in symbols}
        ts_val = ts_index[i]
        ym = str(ts_val.strftime("%Y-%m"))
        deposit_this = 0.0
        if prev_ym is not None and ym != prev_ym and monthly_deposit > 0:
            cash += monthly_deposit
            total_deposited += monthly_deposit
            n_deposits += 1
            deposit_this = monthly_deposit
        prev_ym = ym

        closed: list[str] = []
        opened: list[str] = []
        n_cand = 0

        for sym in list(slots.keys()):
            active_bars[sym] += 1
            slot = slots[sym]
            mid = mids[sym]
            budget = slot.notional_budget
            q = slot.eng.quote(mid, slot.inv, mid_history=mid_window(sym, i))
            if q.quoting:
                if q.bid_size > 0 and rng.random() < _fill_prob(
                    mid, q.bid, q.half_spread_bps, sc.fill_prob
                ):
                    px = q.bid * (1.0 + sc.adverse_frac * (q.half_spread_bps / 1e4))
                    fee = fee_on_notional(px * q.bid_size, cfg)
                    cost = px * q.bid_size + fee
                    if cost <= cash + 1e-9 and (
                        slot.inv.base * mid + cost <= budget * 1.05 + 1e-9
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
                        px = q.ask * (1.0 - sc.adverse_frac * (q.half_spread_bps / 1e4))
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
                n_rt += 1
                closed.append(sym)
                del slots[sym]

        eq_now = mark_equity(mids)
        plan = slot_plan(eq_now)
        max_slots = max(len(plan), 1)
        peak_slots = max(peak_slots, max_slots)
        base_now = slot_size(eq_now)

        if len(slots) >= max_slots:
            idle = [s for s in symbols if s not in slots]
            if idle:
                s = idle[int(rng.integers(0, len(idle)))]
                mid_s = mids[s]
                if mid_s > 0 and np.isfinite(mid_s):
                    bud = plan[-1] if plan else base_now
                    max_inv = max(bud / mid_s, 1e-12)
                    probe = QuoteEngine(cfg.model_copy(update={"max_inventory": max_inv}))
                    flat = InventoryState(
                        base=0.0,
                        quote=0.0,
                        target=0.0,
                        max_abs=max_inv,
                        halt_frac=cfg.inventory_halt_frac,
                    )
                    q_s = probe.quote(mid_s, flat, mid_history=mid_window(s, i))
                    if (
                        q_s.quoting
                        and q_s.net_edge_bps_if_matched > 0
                        and rng.random()
                        < _fill_prob(mid_s, q_s.bid, q_s.half_spread_bps, sc.fill_prob)
                    ):
                        n_busy += 1

        free = max_slots - len(slots)
        if free > 0 and plan:
            used_budgets = sorted(
                (sl.notional_budget for sl in slots.values()), reverse=True
            )
            remaining_plan = list(plan)
            for b in used_budgets:
                for j, p in enumerate(remaining_plan):
                    if abs(p - b) < 1e-6 or (
                        b >= base_now - 1e-6 and p >= base_now - 1e-6
                    ):
                        remaining_plan.pop(j)
                        break

            cands: list[tuple[str, float, Quote]] = []
            probe_budget = remaining_plan[0] if remaining_plan else base_now
            for s in symbols:
                if s in slots:
                    continue
                mid = mids[s]
                if mid <= 0 or not np.isfinite(mid):
                    continue
                max_inv = max(probe_budget / mid, 1e-12)
                local = QuoteEngine(cfg.model_copy(update={"max_inventory": max_inv}))
                flat = InventoryState(
                    base=0.0,
                    quote=0.0,
                    target=0.0,
                    max_abs=max_inv,
                    halt_frac=cfg.inventory_halt_frac,
                )
                q = local.quote(mid, flat, mid_history=mid_window(s, i))
                if (
                    not q.quoting
                    or q.net_edge_bps_if_matched <= 0
                    or q.bid_size <= 0
                ):
                    continue
                if rng.random() < _fill_prob(
                    mid, q.bid, q.half_spread_bps, sc.fill_prob
                ):
                    cands.append((s, _score(q), q))
            n_cand = len(cands)
            if n_cand > 1:
                n_multi += 1
            cands.sort(key=lambda t: t[1], reverse=True)
            for sym, _sc0, _q0 in cands:
                if free <= 0 or not remaining_plan:
                    break
                budget = remaining_plan.pop(0)
                mid = mids[sym]
                max_inv = max(budget / mid, 1e-12)
                local = QuoteEngine(cfg.model_copy(update={"max_inventory": max_inv}))
                flat = InventoryState(
                    base=0.0,
                    quote=0.0,
                    target=0.0,
                    max_abs=max_inv,
                    halt_frac=cfg.inventory_halt_frac,
                )
                q1 = local.quote(mid, flat, mid_history=mid_window(sym, i))
                if not q1.quoting or q1.bid_size <= 0:
                    remaining_plan.insert(0, budget)
                    continue
                px = q1.bid * (1.0 + sc.adverse_frac * (q1.half_spread_bps / 1e4))
                fee = fee_on_notional(px * q1.bid_size, cfg)
                cost = px * q1.bid_size + fee
                if cost > cash + 1e-9:
                    remaining_plan.insert(0, budget)
                    continue
                slot = arm(sym, mid, budget)
                cash -= cost
                prev = slot.inv.realized_pnl
                slot.inv.apply_buy(price=px, size=q1.bid_size, fee=0.0, bar=i)
                fees_total += fee
                realized_total += slot.inv.realized_pnl - prev
                slots[sym] = slot
                n_entries += 1
                n_buys += 1
                opened.append(sym)
                free -= 1

        open_hist.append(len(slots))
        if i % record_every == 0 or i == n - 1 or opened or closed or deposit_this:
            rows.append(
                {
                    "bar": i,
                    "timestamp": ts_val,
                    "cash": cash,
                    "equity": mark_equity(mids),
                    "deposit": deposit_this,
                    "total_deposited": total_deposited,
                    "slot_size": base_now,
                    "n_open_slots": len(slots),
                    "max_slots_allowed": max_slots,
                    "open_symbols": ",".join(sorted(slots.keys())),
                    "opened": ",".join(opened),
                    "closed": ",".join(closed),
                    "n_candidates": n_cand,
                    "fees_cum": fees_total,
                    "realized_pnl": realized_total,
                }
            )

        if i > 0 and i % 10000 == 0:
            print(
                f"  … bar {i}/{n} equity={mark_equity(mids):.2f} "
                f"open={len(slots)} entries={n_entries}",
                flush=True,
            )

    if slots:
        last = {s: float(mid_mat[s][-1]) for s in symbols}
        for sym, slot in list(slots.items()):
            cash += slot.inv.base * last[sym]
            slot.inv.base = 0.0
            n_rt += 1
            del slots[sym]

    final_eq = float(cash)
    net_profit = final_eq - total_deposited
    rod = 100.0 * net_profit / total_deposited if total_deposited else 0.0
    vs_start = (
        100.0 * (final_eq - initial_equity) / initial_equity if initial_equity else 0.0
    )
    active = {k: v for k, v in active_bars.items() if v > 0}
    result = Result(
        n_bars=n,
        n_entries=n_entries,
        n_round_trips=n_rt,
        n_buys=n_buys,
        n_sells=n_sells,
        n_skipped_busy=n_busy,
        n_multi_signal_choices=n_multi,
        peak_slots=peak_slots,
        start_equity=float(initial_equity),
        final_equity=final_eq,
        total_deposited=float(total_deposited),
        net_profit=float(net_profit),
        return_on_deposited_pct=float(rod),
        total_return_vs_start_pct=float(vs_start),
        fees_paid=float(fees_total),
        realized_pnl=float(realized_total),
        n_active_symbols=len(active),
        active_symbol_bars=active_bars,
        symbols=symbols,
        max_open_slots_mean=float(np.mean(open_hist)) if open_hist else 0.0,
        n_deposits=n_deposits,
        interpretation_allowed=False,
        notes=(
            f"Start=${initial_equity:.0f} +${monthly_deposit:.0f}/mo; "
            f"slot ${SLOT_MIN:.0f}→${SLOT_MAX:.0f}, remnant_min=${REMNANT_MIN:.0f}. "
            f"Fee {cfg.commission_pct}%/side. Paper only."
        ),
        open_slots_end=sorted(slots.keys()),
    )
    return result, pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="5m retail MM $250-slot paper forward")
    p.add_argument(
        "--monthly-deposit",
        type=float,
        default=0.0,
        help="USDT added at each month boundary after the first (0 = no DCA)",
    )
    p.add_argument(
        "--out-basename",
        type=str,
        default="",
        help="Output basename under config/ and artifacts/ (auto if empty)",
    )
    args = p.parse_args(argv)
    monthly_deposit = float(args.monthly_deposit)
    if args.out_basename.strip():
        basename = args.out_basename.strip()
    elif monthly_deposit > 0:
        basename = f"forward_5m_250slots_dca{int(monthly_deposit)}_2026"
    else:
        basename = "forward_5m_250slots_2026"

    cfg = load_mm_config("config/market_making_5m.yaml")
    symbols = list(cfg.symbols)
    cfg = cfg.model_copy(update={"portfolio_initial_equity": START_EQUITY})

    dca_note = (
        f"+${monthly_deposit:.0f}/mo DCA"
        if monthly_deposit > 0
        else "no monthly DCA"
    )
    print(
        f"Running 5m $250 start, slots ${SLOT_MIN:.0f}→${SLOT_MAX:.0f} "
        f"({dca_note}, remnant≥${REMNANT_MIN:.0f}) on {len(symbols)} coins "
        f"2026-01-01→2026-09-07 ...",
        flush=True,
    )
    for e in (250, 400, 750, 1250, 2000, 3250):
        print(f"  plan@{e}: size={slot_size(e):.1f} → {slot_plan(e)}", flush=True)

    stamps, panel = load_aligned_panel(
        symbols,
        timeframe="5m",
        start="2026-01-01",
        end_exclusive="2026-09-07",
        root=cfg.storage_root,
        exchange_id=cfg.exchange_id,
    )
    print(f"  loaded {len(stamps)} bars × {len(panel)} symbols", flush=True)
    result, frame = run_dynamic(
        cfg,
        panel,
        stamps,
        seed=42,
        initial_equity=START_EQUITY,
        monthly_deposit=monthly_deposit,
    )

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["ym"] = frame["timestamp"].dt.strftime("%Y-%m")
    monthly = []
    for ym, g in frame.groupby("ym"):
        s0 = float(g.iloc[0]["equity"])
        s1 = float(g.iloc[-1]["equity"])
        dep = float(g["deposit"].sum())
        monthly.append(
            {
                "ym": ym,
                "start": s0,
                "end": s1,
                "deposit": dep,
                "ret_pct": 100.0 * (s1 / s0 - 1.0) if s0 else 0.0,
                "mean_open": float(g["n_open_slots"].mean()),
                "mean_allowed": float(g["max_slots_allowed"].mean()),
                "mean_slot_size": float(g["slot_size"].mean()),
                "total_deposited_end": float(g.iloc[-1]["total_deposited"]),
            }
        )

    out = {
        "interpretation_allowed": False,
        "design": {
            "start_equity": START_EQUITY,
            "monthly_deposit": monthly_deposit,
            "slot_min": SLOT_MIN,
            "slot_max": SLOT_MAX,
            "remnant_min": REMNANT_MIN,
            "slot_max_at_equity": SLOT_MAX_AT_EQUITY,
            "slot_rule": (
                "at $250 → two $100 windows; base slot scales $100→$250 with equity; "
                "remainder>=$100 opens extra remnant window"
            ),
            "deposit_rule": (
                f"+${monthly_deposit:.0f} at each month boundary after start"
                if monthly_deposit > 0
                else "none (compound from $250)"
            ),
            "window": "2026-01-01→2026-09-07",
            "timeframe": "5m",
            "n_symbols_used": len(symbols),
            "params": "params_5m_v1_jfm2025 (locked after Jan–Mar 2025 calib)",
        },
        "result": asdict(result),
        "monthly": monthly,
    }
    Path(f"config/{basename}_results.json").write_text(
        json.dumps(out, indent=2, default=str)
    )
    art = Path("/opt/cursor/artifacts")
    art.mkdir(parents=True, exist_ok=True)
    (art / f"{basename}_results.json").write_text(
        json.dumps(out, indent=2, default=str)
    )
    frame.to_csv(art / f"{basename}_equity.csv", index=False)
    print(
        f"  → start {result.start_equity:.0f} deposited {result.total_deposited:.0f} "
        f"→ final {result.final_equity:.2f} "
        f"(net {result.net_profit:+.2f} / RoD {result.return_on_deposited_pct:.2f}%) "
        f"entries={result.n_entries} busy={result.n_skipped_busy} "
        f"peak_slots={result.peak_slots} deposits={result.n_deposits} "
        f"fees={result.fees_paid:.2f}",
        flush=True,
    )
    for m in monthly:
        print(
            f"  {m['ym']}: {m['start']:.0f}→{m['end']:.0f} "
            f"(dep +{m['deposit']:.0f}, {m['ret_pct']:+.2f}%) "
            f"open≈{m['mean_open']:.1f}/{m['mean_allowed']:.1f} "
            f"slot≈{m['mean_slot_size']:.0f}",
            flush=True,
        )
    print(f"Wrote config/{basename}_results.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
