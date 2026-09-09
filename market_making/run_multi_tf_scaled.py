"""Multi-TF scaled paper MM — $100k plan × 0.1, fee FIXED 0.08%/side.

Layers (scaled): 1m $25 / 30m $100 / 1h $200 / 4h $1000 on $10k equity.
Full plan:       1m $250 / 30m $1000 / 1h $2000 / 4h $10000 on $100k.

Conflict rules
--------------
1. One symbol held by at most one TF at a time.
2. Entry priority: 4h > 1h > 30m > 1m (lower priority cannot steal an open symbol).
3. Per-TF max_slots cap concurrent notionals.
4. Shared USDT cash wallet.

Master clock = 1m OKX mids. Slower layers decide only on their bar closes.
Fee stays at Regular 0.08% (do not apply VIP discount).
interpretation_allowed=false — not a live-edge claim.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from market_making.config import MarketMakingConfig, load_mm_config
from market_making.engine import Quote, QuoteEngine
from market_making.fees import fee_on_notional
from market_making.inventory import InventoryState
from market_making.portfolio import load_aligned_panel
from market_making.simulator import _fill_prob

# Wall-clock matched AS params (from existing locks; 30m/4h derived).
LAYER_PARAMS: dict[str, dict[str, float]] = {
    "1m": {
        "gamma": 0.18,
        "kappa": 0.8,
        "tau_bars": 2880.0,
        "size_eta": 2.5,
        "adverse_vol_mult": 0.35,
        "vol_lookback": 11520.0,
        "ewma_halflife": 5760.0,
        "bars_per_year": 525960.0,
        "decision_every_1m": 1,
    },
    "30m": {
        # From 15m lock ×2 bar length (same ~48h wall tau).
        "gamma": 0.18,
        "kappa": 0.8,
        "tau_bars": 96.0,
        "size_eta": 2.5,
        "adverse_vol_mult": 0.45,
        "vol_lookback": 384.0,
        "ewma_halflife": 192.0,
        "bars_per_year": 17520.0,
        "decision_every_1m": 30,
    },
    "1h": {
        "gamma": 0.18,
        "kappa": 0.8,
        "tau_bars": 48.0,
        "size_eta": 2.5,
        "adverse_vol_mult": 0.45,
        "vol_lookback": 192.0,
        "ewma_halflife": 96.0,
        "bars_per_year": 8760.0,
        "decision_every_1m": 60,
    },
    "4h": {
        # From hourly lock ÷4 bar length (same ~48h wall tau).
        "gamma": 0.18,
        "kappa": 0.8,
        "tau_bars": 12.0,
        "size_eta": 2.5,
        "adverse_vol_mult": 0.45,
        "vol_lookback": 48.0,
        "ewma_halflife": 24.0,
        "bars_per_year": 2190.0,
        "decision_every_1m": 240,
    },
}

PRIORITY = ("4h", "1h", "30m", "1m")


@dataclass
class LayerSlot:
    tf: str
    symbol: str
    inv: InventoryState
    eng: QuoteEngine
    notional_budget: float
    entry_bar: int | None = None


@dataclass
class LayerStats:
    tf: str
    notional: float
    max_slots: int
    n_entries: int = 0
    n_round_trips: int = 0
    n_buys: int = 0
    n_sells: int = 0
    n_blocked_busy_symbol: int = 0
    n_blocked_slot_cap: int = 0
    fees_paid: float = 0.0
    peak_slots: int = 0


@dataclass
class MultiTfResult:
    interpretation_allowed: bool
    commission_pct: float
    scale_factor: float
    start_equity: float
    final_equity: float
    net_profit: float
    total_return_pct: float
    fees_paid: float
    n_bars_1m: int
    n_entries: int
    n_round_trips: int
    n_buys: int
    n_sells: int
    peak_open_total: int
    layers: dict[str, dict[str, Any]]
    symbols: list[str]
    window: dict[str, str]
    full_plan_notionals: dict[str, float]
    scaled_notionals: dict[str, float]
    conflict_rules: list[str]
    notes: str
    open_end: list[str] = field(default_factory=list)


def _score(q: Quote) -> float:
    if not q.quoting:
        return float("-inf")
    return float(q.net_edge_bps_if_matched)


def _cfg_for_layer(
    base: MarketMakingConfig,
    tf: str,
    *,
    commission_pct: float,
    max_inv: float,
) -> MarketMakingConfig:
    p = LAYER_PARAMS[tf]
    return base.model_copy(
        update={
            "commission_pct": commission_pct,
            "gamma": p["gamma"],
            "kappa": p["kappa"],
            "tau_bars": p["tau_bars"],
            "size_eta": p["size_eta"],
            "adverse_vol_mult": p["adverse_vol_mult"],
            "vol_lookback": int(p["vol_lookback"]),
            "ewma_halflife": p["ewma_halflife"],
            "max_inventory": max_inv,
            "inventory_halt_frac": 0.8,
            "liq_bps": 6.0,
            "vol_spread_mult": 0.25,
            "use_as_quotes": True,
            "interpretation_allowed": False,
            "sim": base.sim.model_copy(
                update={
                    "fill_prob": 0.25,
                    "adverse_frac": 0.55,
                    "bars_per_year": p["bars_per_year"],
                    "seed": 42,
                }
            ),
        }
    )


def _is_decision_bar(ts: pd.Timestamp, every_1m: int) -> bool:
    if every_1m <= 1:
        return True
    # Align to UTC minute-of-day buckets.
    mod = (int(ts.hour) * 60 + int(ts.minute)) % int(every_1m)
    return mod == 0


def run_multi_tf(
    *,
    symbols: list[str],
    start: str,
    end_exclusive: str,
    equity0: float,
    commission_pct: float,
    notionals: dict[str, float],
    max_slots: dict[str, int],
    scale_factor: float,
    root: str = "data",
    exchange_id: str = "okx",
    seed: int = 42,
) -> tuple[MultiTfResult, pd.DataFrame]:
    base = load_mm_config("config/multi_tf_scaled_100k_fee008.yaml")
    # Force fee — never VIP-discount in this harness.
    assert abs(commission_pct - 0.08) < 1e-12, "fee must stay 0.08%"

    ts, panel = load_aligned_panel(
        symbols,
        timeframe="1m",
        start=start,
        end_exclusive=end_exclusive,
        root=root,
        exchange_id=exchange_id,
    )
    n = len(ts)
    rng = np.random.default_rng(seed + 91)
    flat_eps = float(base.portfolio_flat_eps)

    cash = float(equity0)
    # slots keyed by symbol (one TF owns a symbol)
    slots: dict[str, LayerSlot] = {}
    hist_1m: dict[str, list[float]] = {s: [] for s in symbols}
    hist_tf: dict[str, dict[str, list[float]]] = {
        tf: {s: [] for s in symbols} for tf in PRIORITY
    }

    stats = {
        tf: LayerStats(
            tf=tf,
            notional=float(notionals[tf]),
            max_slots=int(max_slots[tf]),
        )
        for tf in PRIORITY
    }
    fees_total = 0.0
    n_entries = n_rt = n_buys = n_sells = 0
    peak_open = 0
    rows: list[dict[str, Any]] = []

    def mark_eq(mids: dict[str, float]) -> float:
        eq = cash
        for sym, sl in slots.items():
            eq += sl.inv.base * mids[sym]
        return float(eq)

    def arm(tf: str, sym: str, mid: float, budget: float, bar: int) -> LayerSlot:
        max_inv = max(budget / mid, 1e-12)
        cfg = _cfg_for_layer(base, tf, commission_pct=commission_pct, max_inv=max_inv)
        inv = InventoryState(
            base=0.0,
            quote=0.0,
            target=0.0,
            max_abs=max_inv,
            halt_frac=cfg.inventory_halt_frac,
        )
        return LayerSlot(
            tf=tf,
            symbol=sym,
            inv=inv,
            eng=QuoteEngine(cfg),
            notional_budget=budget,
            entry_bar=bar,
        )

    def tf_open_count(tf: str) -> int:
        return sum(1 for sl in slots.values() if sl.tf == tf)

    for i in range(n):
        mids = {s: float(panel[s][i]) for s in symbols}
        tsi = pd.Timestamp(ts[i])
        for s in symbols:
            hist_1m[s].append(mids[s])

        # Update slower TF histories on their closes.
        for tf in PRIORITY:
            every = int(LAYER_PARAMS[tf]["decision_every_1m"])
            if _is_decision_bar(tsi, every):
                for s in symbols:
                    hist_tf[tf][s].append(mids[s])

        closed: list[str] = []
        opened: list[str] = []

        # Manage open positions every 1m bar (fills).
        for sym in list(slots.keys()):
            sl = slots[sym]
            mid = mids[sym]
            h = np.asarray(hist_tf[sl.tf][sym] or hist_1m[sym], dtype=float)
            q = sl.eng.quote(mid, sl.inv, mid_history=h)
            sc = sl.eng.cfg.sim
            if q.quoting:
                if q.bid_size > 0 and rng.random() < _fill_prob(
                    mid, q.bid, q.half_spread_bps, sc.fill_prob
                ):
                    px = q.bid * (1.0 + sc.adverse_frac * (q.half_spread_bps / 1e4))
                    fee = fee_on_notional(px * q.bid_size, sl.eng.cfg)
                    cost = px * q.bid_size + fee
                    if cost <= cash + 1e-9 and (
                        sl.inv.base * mid + cost <= sl.notional_budget * 1.05 + 1e-9
                    ):
                        cash -= cost
                        prev = sl.inv.realized_pnl
                        sl.inv.apply_buy(price=px, size=q.bid_size, fee=0.0, bar=i)
                        fees_total += fee
                        stats[sl.tf].fees_paid += fee
                        stats[sl.tf].n_buys += 1
                        n_buys += 1
                if q.ask_size > 0 and sl.inv.base > flat_eps:
                    if rng.random() < _fill_prob(
                        mid, q.ask, q.half_spread_bps, sc.fill_prob
                    ):
                        sell_sz = min(q.ask_size, sl.inv.base)
                        px = q.ask * (
                            1.0 - sc.adverse_frac * (q.half_spread_bps / 1e4)
                        )
                        fee = fee_on_notional(px * sell_sz, sl.eng.cfg)
                        cash += px * sell_sz - fee
                        prev = sl.inv.realized_pnl
                        sl.inv.apply_sell(price=px, size=sell_sz, fee=0.0, bar=i)
                        fees_total += fee
                        stats[sl.tf].fees_paid += fee
                        stats[sl.tf].n_sells += 1
                        n_sells += 1
                        _ = prev
            if abs(sl.inv.base) <= flat_eps:
                if abs(sl.inv.base) > 0:
                    cash += sl.inv.base * mid
                    sl.inv.base = 0.0
                stats[sl.tf].n_round_trips += 1
                n_rt += 1
                closed.append(f"{sl.tf}:{sym}")
                del slots[sym]

        # Entries: only on each TF's decision bar, priority order.
        for tf in PRIORITY:
            every = int(LAYER_PARAMS[tf]["decision_every_1m"])
            if not _is_decision_bar(tsi, every):
                continue
            # Need enough TF history to quote.
            if any(len(hist_tf[tf][s]) < 8 for s in symbols):
                continue
            free = int(max_slots[tf]) - tf_open_count(tf)
            if free <= 0:
                stats[tf].n_blocked_slot_cap += 1
                continue
            budget = float(notionals[tf])
            cands: list[tuple[str, float, Quote]] = []
            for s in symbols:
                if s in slots:
                    stats[tf].n_blocked_busy_symbol += 1
                    continue
                mid = mids[s]
                if mid <= 0 or not np.isfinite(mid):
                    continue
                max_inv = max(budget / mid, 1e-12)
                cfg = _cfg_for_layer(
                    base, tf, commission_pct=commission_pct, max_inv=max_inv
                )
                probe = QuoteEngine(cfg)
                flat = InventoryState(
                    base=0.0,
                    quote=0.0,
                    target=0.0,
                    max_abs=max_inv,
                    halt_frac=cfg.inventory_halt_frac,
                )
                h = np.asarray(hist_tf[tf][s], dtype=float)
                q = probe.quote(mid, flat, mid_history=h)
                if (
                    not q.quoting
                    or q.net_edge_bps_if_matched <= 0
                    or q.bid_size <= 0
                ):
                    continue
                if rng.random() < _fill_prob(
                    mid, q.bid, q.half_spread_bps, cfg.sim.fill_prob
                ):
                    cands.append((s, _score(q), q))
            cands.sort(key=lambda t: t[1], reverse=True)
            for sym, _sc0, _q0 in cands:
                if free <= 0:
                    break
                if sym in slots:
                    continue
                mid = mids[sym]
                max_inv = max(budget / mid, 1e-12)
                cfg = _cfg_for_layer(
                    base, tf, commission_pct=commission_pct, max_inv=max_inv
                )
                probe = QuoteEngine(cfg)
                flat = InventoryState(
                    base=0.0,
                    quote=0.0,
                    target=0.0,
                    max_abs=max_inv,
                    halt_frac=cfg.inventory_halt_frac,
                )
                h = np.asarray(hist_tf[tf][sym], dtype=float)
                q1 = probe.quote(mid, flat, mid_history=h)
                if not q1.quoting or q1.bid_size <= 0:
                    continue
                px = q1.bid * (
                    1.0 + cfg.sim.adverse_frac * (q1.half_spread_bps / 1e4)
                )
                fee = fee_on_notional(px * q1.bid_size, cfg)
                cost = px * q1.bid_size + fee
                if cost > cash + 1e-9:
                    continue
                sl = arm(tf, sym, mid, budget, i)
                cash -= cost
                prev = sl.inv.realized_pnl
                sl.inv.apply_buy(price=px, size=q1.bid_size, fee=0.0, bar=i)
                fees_total += fee
                stats[tf].fees_paid += fee
                stats[tf].n_entries += 1
                stats[tf].n_buys += 1
                n_entries += 1
                n_buys += 1
                slots[sym] = sl
                opened.append(f"{tf}:{sym}")
                free -= 1
                _ = prev

        for tf in PRIORITY:
            stats[tf].peak_slots = max(stats[tf].peak_slots, tf_open_count(tf))
        peak_open = max(peak_open, len(slots))
        rows.append(
            {
                "bar": i,
                "timestamp": tsi,
                "cash": cash,
                "equity": mark_eq(mids),
                "n_open": len(slots),
                "open": ",".join(f"{sl.tf}:{sym}" for sym, sl in sorted(slots.items())),
                "opened": ",".join(opened),
                "closed": ",".join(closed),
                "fees_cum": fees_total,
            }
        )

    # Liquidate remainder at last mid.
    if slots:
        last = {s: float(panel[s][-1]) for s in symbols}
        for sym, sl in list(slots.items()):
            cash += sl.inv.base * last[sym]
            sl.inv.base = 0.0
            stats[sl.tf].n_round_trips += 1
            n_rt += 1
            del slots[sym]

    final_eq = float(cash)
    net = final_eq - float(equity0)
    ret = 100.0 * net / float(equity0) if equity0 else 0.0
    result = MultiTfResult(
        interpretation_allowed=False,
        commission_pct=float(commission_pct),
        scale_factor=float(scale_factor),
        start_equity=float(equity0),
        final_equity=final_eq,
        net_profit=float(net),
        total_return_pct=float(ret),
        fees_paid=float(fees_total),
        n_bars_1m=n,
        n_entries=n_entries,
        n_round_trips=n_rt,
        n_buys=n_buys,
        n_sells=n_sells,
        peak_open_total=peak_open,
        layers={tf: asdict(stats[tf]) for tf in PRIORITY},
        symbols=list(symbols),
        window={"start": start, "end_exclusive": end_exclusive},
        full_plan_notionals={tf: float(notionals[tf]) for tf in PRIORITY},
        scaled_notionals={tf: float(notionals[tf]) for tf in PRIORITY},
        conflict_rules=[
            "one_symbol_one_tf",
            "priority_4h>1h>30m>1m",
            "per_tf_max_slots",
            "shared_cash",
            "fee_fixed_0.08_no_vip_discount",
        ],
        notes=(
            f"Equity=${equity0:.0f} scale_factor={scale_factor}; "
            f"notionals 1m/30m/1h/4h="
            f"{notionals['1m']:.0f}/{notionals['30m']:.0f}/"
            f"{notionals['1h']:.0f}/{notionals['4h']:.0f}; "
            f"fee FIXED {commission_pct}%/side (OKX Regular, no VIP). "
            "30m params wall-matched from 15m lock; 4h from hourly lock. "
            "Paper fill_prob=0.25 adverse=0.55. interpretation_allowed=false."
        ),
        open_end=[],
    )
    return result, pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-TF scaled paper @ fee 0.08%")
    p.add_argument(
        "--config",
        default="config/multi_tf_scaled_100k_fee008.yaml",
    )
    p.add_argument(
        "--out",
        default="config/multi_tf_scaled_10k_fee008_augweek_results.json",
    )
    args = p.parse_args()
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    mt = dict(raw.get("multi_tf") or {})
    commission = float(mt.get("commission_pct", 0.08))
    if abs(commission - 0.08) > 1e-12:
        raise SystemExit("Refusing non-0.08 fee — keep Regular 0.08%")

    layers = {L["tf"]: L for L in mt["layers"]}
    notionals = {tf: float(layers[tf]["notional"]) for tf in PRIORITY}
    max_slots = {tf: int(layers[tf]["max_slots"]) for tf in PRIORITY}
    win = mt["window"]
    result, frame = run_multi_tf(
        symbols=list(mt["symbols"]),
        start=str(win["start"]),
        end_exclusive=str(win["end_exclusive"]),
        equity0=float(mt["equity"]),
        commission_pct=commission,
        notionals=notionals,
        max_slots=max_slots,
        scale_factor=float(mt.get("scale_factor", 0.1)),
    )
    out = Path(args.out)
    out.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    pq = out.with_suffix(".parquet")
    frame.to_parquet(pq, index=False)
    print(json.dumps(asdict(result), indent=2))
    print(f"wrote {out} and {pq}", flush=True)


if __name__ == "__main__":
    main()
