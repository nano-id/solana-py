"""Retail MM paper: $4k start, $2k slots, +1 window when remainder >= $500.

Example capacity from mark equity
---------------------------------
$4000 → 2×$2000
$4500 → $2000+$2000+$500
$6500 → 3×$2000+$500
etc.

Window: 2025-09-01 → 2026-09-06 on OKX 1h coins with full coverage.
interpretation_allowed=false.
"""

from __future__ import annotations

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

BASE_SLOT = 2_000.0
REMNANT_MIN = 500.0
START_EQUITY = 4_000.0


def slot_plan(equity: float) -> list[float]:
    """Return per-slot notionals for current mark equity."""
    eq = max(float(equity), 0.0)
    full = int(eq // BASE_SLOT)
    rem = eq - full * BASE_SLOT
    if full <= 0 and rem >= REMNANT_MIN:
        return [rem]
    if rem >= REMNANT_MIN - 1e-9:
        return [BASE_SLOT] * full + [rem]
    return [BASE_SLOT] * max(full, 0)


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
    total_return_pct: float
    fees_paid: float
    realized_pnl: float
    n_active_symbols: int
    active_symbol_bars: dict[str, int]
    symbols: list[str]
    max_open_slots_mean: float
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
) -> tuple[Result, pd.DataFrame]:
    symbols = list(panel.keys())
    n = len(next(iter(panel.values())))
    flat_eps = float(cfg.portfolio_flat_eps)
    sc = cfg.sim
    rng = np.random.default_rng(seed + 77)

    cash = float(initial_equity)
    slots: dict[str, SlotState] = {}
    hist: dict[str, list[float]] = {s: [] for s in symbols}

    n_entries = n_rt = n_buys = n_sells = n_busy = n_multi = 0
    fees_total = realized_total = 0.0
    active_bars = {s: 0 for s in symbols}
    peak_slots = 0
    open_hist: list[int] = []
    rows: list[dict[str, Any]] = []

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

    for i in range(n):
        mids = {s: float(panel[s][i]) for s in symbols}
        for s in symbols:
            hist[s].append(mids[s])
        ts_val = timestamps[i] if timestamps is not None else None
        closed: list[str] = []
        opened: list[str] = []
        n_cand = 0

        # Manage open
        for sym in list(slots.keys()):
            active_bars[sym] += 1
            slot = slots[sym]
            mid = mids[sym]
            budget = slot.notional_budget
            q = slot.eng.quote(mid, slot.inv, mid_history=np.asarray(hist[sym], float))
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

        # Busy skips
        if len(slots) >= max_slots:
            for s in symbols:
                if s in slots:
                    continue
                mid_s = mids[s]
                if mid_s <= 0 or not np.isfinite(mid_s):
                    continue
                bud = plan[-1] if plan else BASE_SLOT
                max_inv = max(bud / mid_s, 1e-12)
                probe = QuoteEngine(cfg.model_copy(update={"max_inventory": max_inv}))
                flat = InventoryState(
                    base=0.0,
                    quote=0.0,
                    target=0.0,
                    max_abs=max_inv,
                    halt_frac=cfg.inventory_halt_frac,
                )
                q_s = probe.quote(mid_s, flat, mid_history=np.asarray(hist[s], float))
                if not q_s.quoting or q_s.net_edge_bps_if_matched <= 0:
                    continue
                if rng.random() < _fill_prob(
                    mid_s, q_s.bid, q_s.half_spread_bps, sc.fill_prob
                ):
                    n_busy += 1

        # Open new slots using next free notional in plan
        free = max_slots - len(slots)
        if free > 0 and plan:
            used_budgets = sorted(
                (sl.notional_budget for sl in slots.values()), reverse=True
            )
            remaining_plan = list(plan)
            # greedily consume plan budgets already used (approx by matching sizes)
            for b in used_budgets:
                for j, p in enumerate(remaining_plan):
                    if abs(p - b) < 1e-6 or (b >= BASE_SLOT - 1e-6 and p >= BASE_SLOT - 1e-6):
                        remaining_plan.pop(j)
                        break

            cands: list[tuple[str, float, Quote]] = []
            probe_budget = remaining_plan[0] if remaining_plan else BASE_SLOT
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
                q = local.quote(mid, flat, mid_history=np.asarray(hist[s], float))
                if not q.quoting or q.net_edge_bps_if_matched <= 0 or q.bid_size <= 0:
                    continue
                if rng.random() < _fill_prob(
                    mid, q.bid, q.half_spread_bps, sc.fill_prob
                ):
                    cands.append((s, _score(q), q))
            n_cand = len(cands)
            if n_cand > 1:
                n_multi += 1
            cands.sort(key=lambda t: t[1], reverse=True)
            for sym, _sc0, q0 in cands:
                if free <= 0 or not remaining_plan:
                    break
                budget = remaining_plan.pop(0)
                mid = mids[sym]
                # rebuild quote sized to this budget
                max_inv = max(budget / mid, 1e-12)
                local = QuoteEngine(cfg.model_copy(update={"max_inventory": max_inv}))
                flat = InventoryState(
                    base=0.0,
                    quote=0.0,
                    target=0.0,
                    max_abs=max_inv,
                    halt_frac=cfg.inventory_halt_frac,
                )
                q1 = local.quote(mid, flat, mid_history=np.asarray(hist[sym], float))
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
        rows.append(
            {
                "bar": i,
                "timestamp": ts_val,
                "cash": cash,
                "equity": mark_equity(mids),
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

    if slots:
        last = {s: float(panel[s][-1]) for s in symbols}
        for sym, slot in list(slots.items()):
            cash += slot.inv.base * last[sym]
            slot.inv.base = 0.0
            n_rt += 1
            del slots[sym]

    final_eq = float(cash)
    ret = 100.0 * (final_eq - initial_equity) / initial_equity if initial_equity else 0.0
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
        total_return_pct=float(ret),
        fees_paid=float(fees_total),
        realized_pnl=float(realized_total),
        n_active_symbols=len(active),
        active_symbol_bars=active_bars,
        symbols=symbols,
        max_open_slots_mean=float(np.mean(open_hist)) if open_hist else 0.0,
        interpretation_allowed=False,
        notes=(
            f"Dynamic slots base=${BASE_SLOT:.0f} remnant_min=${REMNANT_MIN:.0f}. "
            f"Fee {cfg.commission_pct}%/side. Paper only."
        ),
        open_slots_end=sorted(slots.keys()),
    )
    return result, pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Dynamic-slot retail MM paper runner")
    p.add_argument("--equity", type=float, default=START_EQUITY)
    p.add_argument(
        "--tag",
        type=str,
        default="",
        help="Output basename tag, e.g. 4k or 10k (default derived from equity)",
    )
    args = p.parse_args(argv)
    start_eq = float(args.equity)
    tag = args.tag.strip() or (
        f"{int(start_eq/1000)}k" if start_eq >= 1000 and start_eq % 1000 == 0 else f"{int(start_eq)}"
    )
    basename = f"retail_{tag}_step500"

    uni = json.loads(Path("config/universe_100_sep2025_okx.json").read_text())
    symbols = list(uni["symbols"])
    # Retail hourly AS params (same gamma/kappa/tau/fill as 100-coin retail book)
    cfg_path = "config/retail_100coin_35k_v1.yaml"
    cfg = load_mm_config(cfg_path)
    cfg = cfg.model_copy(update={"symbols": symbols, "portfolio_initial_equity": start_eq})

    print(
        f"Running ${start_eq:.0f} step+$500 windows on {len(symbols)} coins "
        f"2025-09-01→2026-09-06 ...",
        flush=True,
    )
    stamps, panel = load_aligned_panel(
        symbols,
        timeframe="1h",
        start="2025-09-01",
        end_exclusive="2026-09-06",
        root=cfg.storage_root,
        exchange_id=cfg.exchange_id,
    )
    result, frame = run_dynamic(cfg, panel, stamps, seed=42, initial_equity=start_eq)

    # monthly
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["ym"] = frame["timestamp"].dt.strftime("%Y-%m")
    monthly = []
    for ym, g in frame.groupby("ym"):
        s0 = float(g.iloc[0]["equity"])
        s1 = float(g.iloc[-1]["equity"])
        monthly.append(
            {
                "ym": ym,
                "start": s0,
                "end": s1,
                "ret_pct": 100.0 * (s1 / s0 - 1.0) if s0 else 0.0,
                "mean_open": float(g["n_open_slots"].mean()),
                "mean_allowed": float(g["max_slots_allowed"].mean()),
            }
        )

    out = {
        "interpretation_allowed": False,
        "design": {
            "start_equity": start_eq,
            "base_slot": BASE_SLOT,
            "remnant_min": REMNANT_MIN,
            "slot_rule": "floor(eq/2000) full $2000 slots; if remainder>=500 open extra window with remainder",
            "window": "2025-09-01→2026-09-06",
            "n_symbols_requested": 100,
            "n_symbols_used": len(symbols),
            "note_symbols": "10 of original top100 lacked Sep-2025 history; ran on 90 with full coverage",
            "params": "retail hourly AS from config",
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
        f"  → {result.start_equity:.0f}→{result.final_equity:.2f} "
        f"({result.total_return_pct:.2f}%) entries={result.n_entries} "
        f"busy={result.n_skipped_busy} peak_slots={result.peak_slots} "
        f"active={result.n_active_symbols}/{len(symbols)} fees={result.fees_paid:.2f}",
        flush=True,
    )
    for m in monthly:
        print(
            f"  {m['ym']}: {m['start']:.0f}→{m['end']:.0f} ({m['ret_pct']:+.2f}%) "
            f"open≈{m['mean_open']:.1f}/{m['mean_allowed']:.1f}",
            flush=True,
        )
    print(f"Wrote config/{basename}_results.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
