"""Live OKX TR 1m MM — grow150 windows, decision @ second 59.

Rules
-----
- Candle clock: act once per 1m bar when sec_in_bar >= 59.0
- Slot plan: grow150 (≤$150 one window; peel $100; remnant ≥$51)
- One symbol → at most one window (no overlap)
- Maker-only (postOnly) limit orders on OKX TR
- Credentials: ~/.config/okx-tr-test.env (OKX_API_KEY/SECRET/PASSWORD)

Does not withdraw. Spot trade only.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ccxt
import numpy as np
import yaml

from market_making.config import load_mm_config
from market_making.engine import QuoteEngine
from market_making.feed import snapshot_from_order_book
from market_making.inventory import InventoryState
from market_making.run_live_paper_200slots import (
    _bar_id,
    _sec_in_bar,
    configure_clock,
    poll_books_once,
)
from market_making.run_retail_5m_250slots import slot_plan_grow150

FLAT_BASE_USD = 0.5  # treat base notional below this as flat dust


@dataclass
class LiveSlot:
    symbol: str
    budget: float
    base: float = 0.0
    avg_cost: float = 0.0
    buy_order_id: str | None = None
    sell_order_id: str | None = None


@dataclass
class BotState:
    slots: dict[str, LiveSlot] = field(default_factory=dict)
    n_decisions: int = 0
    n_buy_orders: int = 0
    n_sell_orders: int = 0
    n_cancels: int = 0
    last_bar_id: int | None = None
    started_ts: float = field(default_factory=time.time)


def _expand(p: str) -> Path:
    return Path(os.path.expanduser(p)).resolve()


def load_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        raise FileNotFoundError(path)
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        out[k.strip()] = v.strip().strip("'").strip('"')
    return out


def connect_okx_tr(env: dict[str, str], hostname: str = "tr.okx.com") -> ccxt.okx:
    ex = ccxt.okx(
        {
            "apiKey": env["OKX_API_KEY"],
            "secret": env["OKX_API_SECRET"],
            "password": env.get("OKX_API_PASSWORD") or env.get("OKX_PASSPHRASE") or "",
            "enableRateLimit": True,
            "hostname": hostname,
            "options": {"defaultType": "spot"},
        }
    )
    ex.load_markets()
    return ex


def usdt_free(balance: dict[str, Any]) -> float:
    fre = (balance.get("free") or {}) if isinstance(balance.get("free"), dict) else {}
    if "USDT" in fre:
        return float(fre["USDT"] or 0.0)
    usdt = (balance.get("USDT") or {}) if isinstance(balance.get("USDT"), dict) else {}
    return float(usdt.get("free") or 0.0)


def base_free(balance: dict[str, Any], symbol: str) -> float:
    base = symbol.split("/")[0]
    fre = (balance.get("free") or {}) if isinstance(balance.get("free"), dict) else {}
    if base in fre:
        return float(fre[base] or 0.0)
    row = (balance.get(base) or {}) if isinstance(balance.get(base), dict) else {}
    return float(row.get("free") or 0.0)


def mark_equity(balance: dict[str, Any], mids: dict[str, float], symbols: list[str]) -> float:
    eq = usdt_free(balance)
    for sym in symbols:
        mid = mids.get(sym)
        if mid and mid > 0:
            eq += base_free(balance, sym) * mid
    return float(eq)


def _score(eng: QuoteEngine, mid: float, hist: list[float], max_inv: float, cfg) -> float:
    inv = InventoryState(base=0.0, quote=0.0, target=0.0, max_abs=max_inv, halt_frac=cfg.inventory_halt_frac)
    local = QuoteEngine(cfg.model_copy(update={"max_inventory": max_inv}))
    q = local.quote(mid, inv, mid_history=np.asarray(hist, dtype=float))
    if not q.quoting or q.bid_size <= 0:
        return float("-inf")
    return float(q.net_edge_bps_if_matched)


def cancel_slot_orders(ex: ccxt.Exchange, slot: LiveSlot) -> int:
    n = 0
    for oid in (slot.buy_order_id, slot.sell_order_id):
        if not oid:
            continue
        try:
            ex.cancel_order(oid, slot.symbol)
            n += 1
        except Exception:
            pass
    slot.buy_order_id = None
    slot.sell_order_id = None
    return n


def place_postonly(
    ex: ccxt.Exchange,
    symbol: str,
    side: str,
    amount: float,
    price: float,
) -> str | None:
    amount = float(ex.amount_to_precision(symbol, amount))
    price = float(ex.price_to_precision(symbol, price))
    if amount <= 0 or price <= 0:
        return None
    try:
        # OKX spot post-only
        order = ex.create_order(
            symbol,
            "limit",
            side,
            amount,
            price,
            params={"postOnly": True},
        )
        return str(order.get("id") or "")
    except Exception as exc:
        print(f"  order fail {side} {symbol}: {exc}", flush=True)
        return None


def save_state(path: Path, state: BotState, extra: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_decisions": state.n_decisions,
        "n_buy_orders": state.n_buy_orders,
        "n_sell_orders": state.n_sell_orders,
        "n_cancels": state.n_cancels,
        "last_bar_id": state.last_bar_id,
        "started_ts": state.started_ts,
        "slots": {
            s: {
                "symbol": sl.symbol,
                "budget": sl.budget,
                "base": sl.base,
                "avg_cost": sl.avg_cost,
                "buy_order_id": sl.buy_order_id,
                "sell_order_id": sl.sell_order_id,
            }
            for s, sl in state.slots.items()
        },
        **extra,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_bot(
    *,
    config_path: str,
    duration_sec: float = 0.0,
) -> None:
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    live = dict(raw.get("live") or {})
    cfg = load_mm_config(config_path)
    symbols = list(cfg.symbols)
    bar_sec = int(live.get("bar_sec", 60))
    decision_sec = float(live.get("decision_sec", 59.0))
    poll_sec = float(live.get("poll_sec", 0.2))
    hostname = str((raw.get("exchange") or {}).get("hostname") or "tr.okx.com")
    env_path = _expand(str(live.get("env_file", "~/.config/okx-tr-test.env")))
    log_dir = _expand(str(live.get("log_dir", "~/bot/logs")))
    state_path = log_dir / "live_okx_tr_1m_grow150_state.json"
    event_path = log_dir / "live_okx_tr_1m_grow150_events.jsonl"

    configure_clock(bar_sec, decision_sec)
    env = load_env_file(env_path)
    ex = connect_okx_tr(env, hostname=hostname)
    # Restrict to markets that exist on this venue.
    symbols = [s for s in symbols if s in ex.markets]
    print(
        f"LIVE OKX TR 1m grow150 | symbols={len(symbols)} decision>={decision_sec} "
        f"host={hostname} duration={duration_sec or '∞'}",
        flush=True,
    )

    state = BotState()
    hist: dict[str, list[float]] = {s: [] for s in symbols}
    t0 = time.time()
    last_decided: int | None = None

    while True:
        if duration_sec > 0 and (time.time() - t0) >= duration_sec:
            print("duration reached — stopping", flush=True)
            break
        now_ms = int(time.time() * 1000)
        bar = _bar_id(now_ms)
        sec = _sec_in_bar(now_ms)

        if sec + 1e-9 >= decision_sec and bar != last_decided:
            try:
                books = poll_books_once(ex, symbols, exchange_id="okx")
                mids = {s: b.mid for s, b in books.items() if b and b.valid}
                for s, mid in mids.items():
                    hist.setdefault(s, []).append(mid)
                    if len(hist[s]) > max(int(cfg.vol_lookback), 240):
                        del hist[s][: len(hist[s]) - max(int(cfg.vol_lookback), 240)]

                bal = ex.fetch_balance()
                equity = mark_equity(bal, mids, symbols)
                plan = slot_plan_grow150(
                    equity,
                    grow_max=float(live.get("grow_max", 150)),
                    peel=float(live.get("peel", 100)),
                    remnant_min=float(live.get("remnant_min", 51)),
                )
                max_slots = max(len(plan), 1)

                # Cancel resting orders before re-quoting.
                for sl in list(state.slots.values()):
                    state.n_cancels += cancel_slot_orders(ex, sl)

                # Sync bases from exchange; drop flat slots.
                for sym in list(state.slots.keys()):
                    mid = mids.get(sym, 0.0)
                    base = base_free(bal, sym)
                    state.slots[sym].base = base
                    if mid > 0 and base * mid < FLAT_BASE_USD:
                        state.slots[sym].base = 0.0
                        # keep slot record until we decide to free for re-entry
                        if not state.slots[sym].buy_order_id and not state.slots[sym].sell_order_id:
                            # flat — free the symbol slot
                            del state.slots[sym]

                # Exit quotes for open inventory (maker sell).
                for sym, sl in list(state.slots.items()):
                    mid = mids.get(sym)
                    book = books.get(sym)
                    if mid is None or book is None or sl.base * mid < FLAT_BASE_USD:
                        continue
                    max_inv = max(sl.budget / mid, 1e-12)
                    eng = QuoteEngine(cfg.model_copy(update={"max_inventory": max_inv}))
                    inv = InventoryState(
                        base=sl.base,
                        quote=0.0,
                        target=0.0,
                        max_abs=max_inv,
                        halt_frac=cfg.inventory_halt_frac,
                        avg_cost=sl.avg_cost,
                    )
                    q = eng.quote(mid, inv, mid_history=np.asarray(hist.get(sym, [mid]), float))
                    if q.quoting and q.ask_size > 0:
                        px = min(q.ask, book.ask) if book.ask > 0 else q.ask
                        # stay maker: at or above ask
                        px = max(px, book.ask)
                        amt = min(sl.base, q.ask_size)
                        oid = place_postonly(ex, sym, "sell", amt, px)
                        if oid:
                            sl.sell_order_id = oid
                            state.n_sell_orders += 1

                # Entries: fill free slots from plan with best flat symbols.
                free = max_slots - len(state.slots)
                if free > 0 and plan:
                    used = sorted((sl.budget for sl in state.slots.values()), reverse=True)
                    rem_plan = list(plan)
                    for b in used:
                        for j, p in enumerate(rem_plan):
                            if abs(p - b) < 1e-6:
                                rem_plan.pop(j)
                                break
                        else:
                            if rem_plan:
                                rem_plan.pop(0)

                    cash = usdt_free(bal)
                    cands: list[tuple[str, float]] = []
                    for s in symbols:
                        if s in state.slots:
                            continue
                        mid = mids.get(s)
                        if not mid or mid <= 0 or s not in books:
                            continue
                        if len(hist.get(s, [])) < 8:
                            continue
                        # already holding dust from prior rounds — skip to avoid clash
                        if base_free(bal, s) * mid >= FLAT_BASE_USD:
                            continue
                        bud = rem_plan[0] if rem_plan else plan[-1]
                        max_inv = max(bud / mid, 1e-12)
                        sc = _score(QuoteEngine(cfg), mid, hist[s], max_inv, cfg)
                        if sc > 0:
                            cands.append((s, sc))
                    cands.sort(key=lambda t: t[1], reverse=True)

                    for sym, _sc in cands:
                        if free <= 0 or not rem_plan:
                            break
                        if cash < 5:
                            break
                        budget = float(rem_plan.pop(0))
                        budget = min(budget, cash * 0.98)
                        mid = mids[sym]
                        book = books[sym]
                        max_inv = max(budget / mid, 1e-12)
                        eng = QuoteEngine(cfg.model_copy(update={"max_inventory": max_inv}))
                        inv = InventoryState(
                            base=0.0,
                            quote=0.0,
                            target=0.0,
                            max_abs=max_inv,
                            halt_frac=cfg.inventory_halt_frac,
                        )
                        q = eng.quote(
                            mid, inv, mid_history=np.asarray(hist[sym], float)
                        )
                        if not q.quoting or q.bid_size <= 0:
                            rem_plan.insert(0, budget)
                            continue
                        px = min(q.bid, book.bid)
                        notional = min(budget, q.bid * q.bid_size)
                        amt = notional / px if px > 0 else 0.0
                        oid = place_postonly(ex, sym, "buy", amt, px)
                        if oid:
                            state.slots[sym] = LiveSlot(
                                symbol=sym, budget=budget, buy_order_id=oid
                            )
                            state.n_buy_orders += 1
                            cash -= notional
                            free -= 1
                            print(
                                f"  BUY postOnly {sym} px={px} amt≈{amt:.6g} bud={budget:.2f}",
                                flush=True,
                            )

                state.n_decisions += 1
                state.last_bar_id = bar
                last_decided = bar
                extra = {
                    "equity": equity,
                    "usdt_free": usdt_free(bal),
                    "plan": plan,
                    "open_symbols": sorted(state.slots.keys()),
                    "sec_in_bar": sec,
                    "ts": time.time(),
                }
                save_state(state_path, state, extra)
                with event_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(extra) + "\n")
                print(
                    f"decision bar={bar} sec={sec:.3f} eq={equity:.2f} "
                    f"plan={plan} open={sorted(state.slots.keys())} "
                    f"buys={state.n_buy_orders} sells={state.n_sell_orders}",
                    flush=True,
                )
            except Exception:
                print("decision error:\n" + traceback.format_exc(), flush=True)
                last_decided = bar

        # sleep schedule
        now_ms = int(time.time() * 1000)
        sec = _sec_in_bar(now_ms)
        if sec + 1e-9 < decision_sec:
            remain = decision_sec - sec
            wait = 0.05 if remain <= 1.0 else min(poll_sec, max(0.05, remain - 0.2))
        else:
            wait = min(poll_sec, max(0.05, bar_sec - sec))
        if duration_sec > 0:
            rem = duration_sec - (time.time() - t0)
            if rem <= 0:
                break
            wait = min(wait, rem)
        time.sleep(wait)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Live OKX TR 1m grow150 MM")
    p.add_argument("--config", default="config/live_okx_tr_1m_grow150.yaml")
    p.add_argument(
        "--duration-sec",
        type=float,
        default=0.0,
        help="0 = run forever",
    )
    args = p.parse_args(argv)
    run_bot(config_path=args.config, duration_sec=float(args.duration_sec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
