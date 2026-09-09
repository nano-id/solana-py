"""Deterministic 1m OHLC-touch paper for grow150 live rules.

No RNG / no Bernoulli fills. Resting maker limits fill only when the next
candles' low/high touch the quote:

- Decision at each 1m close (proxy for sec=59).
- BUY placed at AS bid; held fixed up to ``buy_timeout_bars`` (default 2 ≈ 120s).
- BUY fills if low <= bid; else TIMEOUT (= stop) and free the window.
- After inventory: SELL resting ask refreshed each bar; fills if high >= ask
  (= target / round-trip complete).
- Fee = commission_pct per side at fill price.

interpretation_allowed=false — hist OHLC touch proxy, not live L2.
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
from market_making.engine import QuoteEngine
from market_making.fees import fee_on_notional
from market_making.inventory import InventoryState
from market_making.run_retail_5m_250slots import slot_plan_grow150


@dataclass
class PendingBuy:
    symbol: str
    px: float
    size: float
    budget: float
    placed_i: int
    eng: QuoteEngine
    inv: InventoryState


@dataclass
class OpenPos:
    symbol: str
    budget: float
    eng: QuoteEngine
    inv: InventoryState
    sell_px: float
    entry_i: int


@dataclass
class Result:
    n_bars_trade: int
    n_buy_orders: int
    n_buy_fills: int
    n_buy_stops: int
    n_targets: int
    n_open_eod: int
    n_sell_fills: int
    fees_paid: float
    realized_pnl: float
    start_equity: float
    final_equity: float
    return_pct: float
    cash_end: float
    inventory_mark_end: float
    interpretation_allowed: bool
    notes: str
    stops: list[dict[str, Any]] = field(default_factory=list)
    targets: list[dict[str, Any]] = field(default_factory=list)
    open_eod: list[dict[str, Any]] = field(default_factory=list)


def load_ohlc_panel(
    symbols: list[str],
    *,
    timeframe: str,
    start: str,
    end_exclusive: str,
    root: str | Path,
    exchange_id: str,
) -> tuple[pd.DatetimeIndex, dict[str, pd.DataFrame]]:
    frames: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        base = sym.replace("/", "_")
        path = Path(root) / exchange_id / timeframe / f"{base}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_parquet(path)
        ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        mask = (ts >= pd.Timestamp(start, tz="UTC")) & (
            ts < pd.Timestamp(end_exclusive, tz="UTC")
        )
        sl = df.loc[mask].copy()
        sl.index = ts.loc[mask]
        sl = sl.sort_index()
        need = {"open", "high", "low", "close"}
        if not need.issubset(set(sl.columns)):
            raise ValueError(f"{path} missing OHLC columns")
        frames[sym] = sl[["open", "high", "low", "close"]].astype(float)
    # inner join on timestamp
    idx = None
    for sym, fr in frames.items():
        idx = fr.index if idx is None else idx.intersection(fr.index)
    if idx is None or len(idx) < 2:
        raise ValueError("OHLC panel too short after align")
    idx = pd.DatetimeIndex(idx).sort_values()
    out = {sym: frames[sym].loc[idx] for sym in symbols}
    return idx, out


def _score(q) -> float:
    if not q.quoting:
        return float("-inf")
    return float(q.net_edge_bps_if_matched)


def run_touch(
    cfg,
    ohlc: dict[str, pd.DataFrame],
    timestamps: pd.DatetimeIndex,
    *,
    trade_start: str,
    trade_end_exclusive: str,
    initial_equity: float = 60.0,
    buy_timeout_bars: int = 2,
) -> Result:
    symbols = list(ohlc.keys())
    n = len(timestamps)
    closes = {s: ohlc[s]["close"].to_numpy(dtype=float) for s in symbols}
    highs = {s: ohlc[s]["high"].to_numpy(dtype=float) for s in symbols}
    lows = {s: ohlc[s]["low"].to_numpy(dtype=float) for s in symbols}
    look = int(cfg.vol_lookback)
    flat_eps = float(cfg.portfolio_flat_eps)

    t0 = pd.Timestamp(trade_start, tz="UTC")
    t1 = pd.Timestamp(trade_end_exclusive, tz="UTC")
    trade_mask = np.asarray((timestamps >= t0) & (timestamps < t1), dtype=bool)
    trade_idx = np.flatnonzero(trade_mask)
    if len(trade_idx) == 0:
        raise ValueError(f"No bars in trade window {trade_start}→{trade_end_exclusive}")

    cash = float(initial_equity)
    pending: PendingBuy | None = None
    pos: OpenPos | None = None

    n_buy_orders = n_buy_fills = n_buy_stops = n_targets = n_sell_fills = 0
    fees_total = realized_total = 0.0
    stops: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    open_eod: list[dict[str, Any]] = []

    def mid_hist(sym: str, i: int) -> np.ndarray:
        lo = 0 if i + 1 <= look else i + 1 - look
        return closes[sym][lo : i + 1]

    def mark(i: int) -> float:
        eq = cash
        if pos is not None:
            eq += pos.inv.base * float(closes[pos.symbol][i])
        return float(eq)

    for i in trade_idx:
        mids = {s: float(closes[s][i]) for s in symbols}
        ts = timestamps[i]

        # --- resting BUY vs this bar's low ---
        if pending is not None:
            pb = pending
            lo = float(lows[pb.symbol][i])
            if lo <= pb.px + 1e-12:
                fee = fee_on_notional(pb.px * pb.size, cfg)
                cost = pb.px * pb.size + fee
                if cost <= cash + 1e-9:
                    cash -= cost
                    prev = pb.inv.realized_pnl
                    pb.inv.apply_buy(price=pb.px, size=pb.size, fee=0.0, bar=i)
                    fees_total += fee
                    realized_total += pb.inv.realized_pnl - prev
                    n_buy_fills += 1
                    q_ex = pb.eng.quote(
                        mids[pb.symbol], pb.inv, mid_history=mid_hist(pb.symbol, i)
                    )
                    sell_px = float(q_ex.ask) if q_ex.quoting and q_ex.ask_size > 0 else pb.px * 1.002
                    pos = OpenPos(
                        symbol=pb.symbol,
                        budget=pb.budget,
                        eng=pb.eng,
                        inv=pb.inv,
                        sell_px=sell_px,
                        entry_i=i,
                    )
                    pending = None
                else:
                    # cannot afford — treat as cancel/stop
                    n_buy_stops += 1
                    stops.append(
                        {
                            "ts": str(ts),
                            "symbol": pb.symbol,
                            "reason": "insufficient_cash",
                            "buy_px": pb.px,
                        }
                    )
                    pending = None
            elif (i - pb.placed_i) >= buy_timeout_bars:
                n_buy_stops += 1
                stops.append(
                    {
                        "ts": str(ts),
                        "symbol": pb.symbol,
                        "reason": "buy_timeout",
                        "buy_px": pb.px,
                        "bars_waited": int(i - pb.placed_i),
                        "low_min": float(
                            np.min(lows[pb.symbol][pb.placed_i + 1 : i + 1])
                        )
                        if i > pb.placed_i
                        else lo,
                    }
                )
                pending = None

        # --- resting SELL vs this bar's high ---
        if pos is not None:
            op = pos
            hi = float(highs[op.symbol][i])
            # only bars after entry can hit the resting ask from prior decision
            if i > op.entry_i and hi >= op.sell_px - 1e-12 and op.inv.base > flat_eps:
                sell_sz = float(op.inv.base)
                px = float(op.sell_px)
                fee = fee_on_notional(px * sell_sz, cfg)
                cash += px * sell_sz - fee
                prev = op.inv.realized_pnl
                op.inv.apply_sell(price=px, size=sell_sz, fee=0.0, bar=i)
                fees_total += fee
                realized_total += op.inv.realized_pnl - prev
                n_sell_fills += 1
                n_targets += 1
                targets.append(
                    {
                        "ts": str(ts),
                        "symbol": op.symbol,
                        "sell_px": px,
                        "bars_held": int(i - op.entry_i),
                        "realized_pnl": float(op.inv.realized_pnl),
                    }
                )
                pos = None
            else:
                # refresh ask for subsequent bars (live re-quotes exit)
                q = op.eng.quote(
                    mids[op.symbol], op.inv, mid_history=mid_hist(op.symbol, i)
                )
                if q.quoting and q.ask_size > 0:
                    op.sell_px = float(q.ask)

        # --- new BUY only if window free ---
        if pending is None and pos is None:
            eq_now = mark(i)
            plan = slot_plan_grow150(eq_now, grow_max=150.0, peel=100.0, remnant_min=51.0)
            if plan:
                budget = float(plan[0])
                cands: list[tuple[str, float, Any]] = []
                for s in symbols:
                    mid = mids[s]
                    if mid <= 0 or not np.isfinite(mid):
                        continue
                    max_inv = max(budget / mid, 1e-12)
                    eng = QuoteEngine(cfg.model_copy(update={"max_inventory": max_inv}))
                    flat = InventoryState(
                        base=0.0,
                        quote=0.0,
                        target=0.0,
                        max_abs=max_inv,
                        halt_frac=cfg.inventory_halt_frac,
                    )
                    q = eng.quote(mid, flat, mid_history=mid_hist(s, i))
                    if (
                        not q.quoting
                        or q.net_edge_bps_if_matched <= 0
                        or q.bid_size <= 0
                    ):
                        continue
                    cands.append((s, _score(q), q))
                cands.sort(key=lambda t: t[1], reverse=True)
                if cands:
                    sym, _sc, q0 = cands[0]
                    mid = mids[sym]
                    max_inv = max(budget / mid, 1e-12)
                    eng = QuoteEngine(cfg.model_copy(update={"max_inventory": max_inv}))
                    inv = InventoryState(
                        base=0.0,
                        quote=0.0,
                        target=0.0,
                        max_abs=max_inv,
                        halt_frac=cfg.inventory_halt_frac,
                    )
                    q1 = eng.quote(mid, inv, mid_history=mid_hist(sym, i))
                    size = float(q1.bid_size)
                    # cap notional to budget / cash
                    max_sz_budget = budget / q1.bid if q1.bid > 0 else 0.0
                    max_sz_cash = (cash * 0.999) / (q1.bid * (1.0 + cfg.commission_frac))
                    size = min(size, max_sz_budget, max_sz_cash)
                    if size > 0 and q1.bid > 0:
                        pending = PendingBuy(
                            symbol=sym,
                            px=float(q1.bid),
                            size=size,
                            budget=budget,
                            placed_i=int(i),
                            eng=eng,
                            inv=inv,
                        )
                        n_buy_orders += 1

    # EOD flatten mark (no forced sell — report open)
    last_i = int(trade_idx[-1])
    if pending is not None:
        n_buy_stops += 1
        stops.append(
            {
                "ts": str(timestamps[last_i]),
                "symbol": pending.symbol,
                "reason": "eod_unfilled_buy",
                "buy_px": pending.px,
            }
        )
        pending = None
    if pos is not None:
        mid = float(closes[pos.symbol][last_i])
        open_eod.append(
            {
                "ts": str(timestamps[last_i]),
                "symbol": pos.symbol,
                "base": float(pos.inv.base),
                "avg_cost": float(pos.inv.avg_cost) if hasattr(pos.inv, "avg_cost") else None,
                "mid": mid,
                "sell_px": float(pos.sell_px),
                "mark_pnl": float(pos.inv.base * (mid - getattr(pos.inv, "avg_cost", mid))),
            }
        )
        n_open = 1
    else:
        n_open = 0

    inv_mark = 0.0
    if pos is not None:
        inv_mark = float(pos.inv.base * closes[pos.symbol][last_i])
    final_eq = cash + inv_mark

    return Result(
        n_bars_trade=int(len(trade_idx)),
        n_buy_orders=n_buy_orders,
        n_buy_fills=n_buy_fills,
        n_buy_stops=n_buy_stops,
        n_targets=n_targets,
        n_open_eod=n_open,
        n_sell_fills=n_sell_fills,
        fees_paid=float(fees_total),
        realized_pnl=float(realized_total),
        start_equity=float(initial_equity),
        final_equity=float(final_eq),
        return_pct=float(100.0 * (final_eq / initial_equity - 1.0)),
        cash_end=float(cash),
        inventory_mark_end=float(inv_mark),
        interpretation_allowed=False,
        notes=(
            "Deterministic OHLC touch: buy if low<=bid within "
            f"{buy_timeout_bars} bars else stop; target if high>=ask. "
            "No RNG. Fee per side at fill. Not a live-edge claim."
        ),
        stops=stops,
        targets=targets,
        open_eod=open_eod,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config/live_okx_tr_1m_grow150.yaml")
    p.add_argument("--equity", type=float, default=60.0)
    p.add_argument("--warmup-start", default="2026-08-17")
    p.add_argument("--trade-start", default="2026-09-01")
    p.add_argument("--trade-end", default="2026-09-02")
    p.add_argument("--buy-timeout-bars", type=int, default=2)
    p.add_argument(
        "--out",
        default="config/ohlc_touch_1m_grow150_sep1_2026_results.json",
    )
    p.add_argument(
        "--symbols",
        default="",
        help="Comma list; default = config symbols (30). Use 12-coin set via flag.",
    )
    p.add_argument("--twelve", action="store_true", help="Use prior grow150 12-coin set")
    args = p.parse_args(argv)

    cfg = load_mm_config(args.config)
    twelve = [
        "BTC/USDT",
        "ETH/USDT",
        "BNB/USDT",
        "SOL/USDT",
        "XRP/USDT",
        "DOGE/USDT",
        "ADA/USDT",
        "TRX/USDT",
        "AVAX/USDT",
        "LINK/USDT",
        "DOT/USDT",
        "LTC/USDT",
    ]
    if args.symbols.strip():
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    elif args.twelve:
        symbols = twelve
    else:
        symbols = list(cfg.symbols)

    stamps, panel = load_ohlc_panel(
        symbols,
        timeframe="1m",
        start=args.warmup_start,
        end_exclusive=args.trade_end,
        root=cfg.storage_root,
        exchange_id=cfg.exchange_id,
    )
    # cfg may use nested live yaml — load_mm_config should expose market_making fields
    res = run_touch(
        cfg,
        panel,
        stamps,
        trade_start=args.trade_start,
        trade_end_exclusive=args.trade_end,
        initial_equity=float(args.equity),
        buy_timeout_bars=int(args.buy_timeout_bars),
    )
    payload = {
        "interpretation_allowed": False,
        "design": {
            "mode": "ohlc_touch_deterministic",
            "start_equity": args.equity,
            "slot_mode": "grow150",
            "buy_timeout_bars": args.buy_timeout_bars,
            "trade_window": f"{args.trade_start}→{args.trade_end}",
            "warmup_start": args.warmup_start,
            "n_symbols": len(symbols),
            "symbols": symbols,
            "fill_model": "low<=bid / high>=ask; no RNG; no adverse_frac",
            "decision": "1m close (== end-of-bar)",
            "params": args.config,
        },
        "result": asdict(res),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["result"], indent=2), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
