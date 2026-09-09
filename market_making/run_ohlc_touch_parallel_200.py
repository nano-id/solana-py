"""Deterministic OHLC-touch paper: fat book, $200 clips, no timeout.

- $10M book / $200 per position → capital never blocks another symbol.
- One state per symbol (pending buy OR inventory); symbols independent.
- AS bid/ask from 1m close; fill only when a later candle low/high touches.
- No Bernoulli fill_prob. No 2-minute buy timeout / stop.
- Pending buy rests until fill or end of window; exit ask refreshed each bar.
- Primary metric: n_targets (round-trips that hit the ask).

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


TWELVE = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT",
    "ADA/USDT", "TRX/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT", "LTC/USDT",
]


@dataclass
class PendingBuy:
    symbol: str
    px: float
    size: float
    notional: float
    placed_i: int
    eng: QuoteEngine
    inv: InventoryState


@dataclass
class OpenPos:
    symbol: str
    notional: float
    eng: QuoteEngine
    inv: InventoryState
    sell_px: float
    entry_i: int
    buy_px: float


@dataclass
class Result:
    n_bars_trade: int
    n_buy_orders: int
    n_buy_fills: int
    n_targets: int
    n_pending_eod: int
    n_open_eod: int
    n_sell_fills: int
    fees_paid: float
    realized_pnl: float
    start_equity: float
    final_equity: float
    return_pct: float
    cash_end: float
    inventory_mark_end: float
    avg_bars_to_buy_fill: float | None
    avg_bars_held_to_target: float | None
    interpretation_allowed: bool
    notes: str
    targets_by_symbol: dict[str, int] = field(default_factory=dict)
    targets_sample: list[dict[str, Any]] = field(default_factory=list)
    open_eod_sample: list[dict[str, Any]] = field(default_factory=list)
    pending_eod_sample: list[dict[str, Any]] = field(default_factory=list)


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
        path = Path(root) / exchange_id / timeframe / f"{sym.replace('/', '_')}.parquet"
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
        frames[sym] = sl[["open", "high", "low", "close"]].astype(float)
    idx: pd.DatetimeIndex | None = None
    for fr in frames.values():
        idx = fr.index if idx is None else idx.intersection(fr.index)
    if idx is None or len(idx) < 2:
        raise ValueError("OHLC panel too short after align")
    idx = pd.DatetimeIndex(idx).sort_values()
    return idx, {sym: frames[sym].loc[idx] for sym in symbols}


def run_parallel_touch(
    cfg,
    ohlc: dict[str, pd.DataFrame],
    timestamps: pd.DatetimeIndex,
    *,
    trade_start: str,
    trade_end_exclusive: str,
    initial_equity: float,
    per_position_usd: float,
) -> Result:
    symbols = list(ohlc.keys())
    closes = {s: ohlc[s]["close"].to_numpy(dtype=float) for s in symbols}
    highs = {s: ohlc[s]["high"].to_numpy(dtype=float) for s in symbols}
    lows = {s: ohlc[s]["low"].to_numpy(dtype=float) for s in symbols}
    look = int(cfg.vol_lookback)
    flat_eps = float(cfg.portfolio_flat_eps)

    t0 = pd.Timestamp(trade_start, tz="UTC")
    t1 = pd.Timestamp(trade_end_exclusive, tz="UTC")
    trade_idx = np.flatnonzero(np.asarray((timestamps >= t0) & (timestamps < t1)))
    if len(trade_idx) == 0:
        raise ValueError(f"No bars in {trade_start}->{trade_end_exclusive}")

    cash = float(initial_equity)
    pending: dict[str, PendingBuy] = {}
    positions: dict[str, OpenPos] = {}

    n_buy_orders = n_buy_fills = n_targets = n_sell_fills = 0
    fees_total = realized_total = 0.0
    buy_wait_bars: list[int] = []
    hold_bars: list[int] = []
    target_rows: list[dict[str, Any]] = []
    targets_by_symbol: dict[str, int] = {}

    def mid_hist(sym: str, i: int) -> np.ndarray:
        lo = 0 if i + 1 <= look else i + 1 - look
        return closes[sym][lo : i + 1]

    def inv_mark(i: int) -> float:
        return float(
            sum(op.inv.base * float(closes[sym][i]) for sym, op in positions.items())
        )

    max_concurrent = max(int(initial_equity // per_position_usd), 1)

    for i in trade_idx:
        mids = {s: float(closes[s][i]) for s in symbols}
        ts = timestamps[i]

        # Resting buys vs this bar low — no timeout; miss → reprice from this close.
        for sym in list(pending.keys()):
            pb = pending[sym]
            if float(lows[sym][i]) > pb.px + 1e-12:
                q_re = pb.eng.quote(mids[sym], pb.inv, mid_history=mid_hist(sym, i))
                if q_re.quoting and q_re.bid_size > 0 and q_re.bid > 0:
                    pb.px = float(q_re.bid)
                    size = min(float(q_re.bid_size), pb.notional / pb.px)
                    size = min(size, (cash * 0.999) / (pb.px * (1.0 + cfg.commission_frac)))
                    if size > 0:
                        pb.size = float(size)
                continue
            fee = fee_on_notional(pb.px * pb.size, cfg)
            cost = pb.px * pb.size + fee
            if cost > cash + 1e-9:
                continue
            cash -= cost
            prev = pb.inv.realized_pnl
            pb.inv.apply_buy(price=pb.px, size=pb.size, fee=0.0, bar=int(i))
            fees_total += fee
            realized_total += pb.inv.realized_pnl - prev
            n_buy_fills += 1
            buy_wait_bars.append(int(i - pb.placed_i))
            q_ex = pb.eng.quote(mids[sym], pb.inv, mid_history=mid_hist(sym, i))
            sell_px = (
                float(q_ex.ask)
                if q_ex.quoting and q_ex.ask_size > 0
                else pb.px * 1.002
            )
            positions[sym] = OpenPos(
                symbol=sym,
                notional=pb.notional,
                eng=pb.eng,
                inv=pb.inv,
                sell_px=sell_px,
                entry_i=int(i),
                buy_px=pb.px,
            )
            del pending[sym]

        # Resting sells vs this bar high; else refresh ask.
        for sym in list(positions.keys()):
            op = positions[sym]
            hi = float(highs[sym][i])
            if i > op.entry_i and hi >= op.sell_px - 1e-12 and op.inv.base > flat_eps:
                sell_sz = float(op.inv.base)
                px = float(op.sell_px)
                fee = fee_on_notional(px * sell_sz, cfg)
                cash += px * sell_sz - fee
                prev = op.inv.realized_pnl
                op.inv.apply_sell(price=px, size=sell_sz, fee=0.0, bar=int(i))
                fees_total += fee
                realized_total += op.inv.realized_pnl - prev
                n_sell_fills += 1
                n_targets += 1
                targets_by_symbol[sym] = targets_by_symbol.get(sym, 0) + 1
                held = int(i - op.entry_i)
                hold_bars.append(held)
                target_rows.append(
                    {
                        "ts": str(ts),
                        "symbol": sym,
                        "buy_px": op.buy_px,
                        "sell_px": px,
                        "bars_held": held,
                        "realized_pnl": float(op.inv.realized_pnl),
                    }
                )
                del positions[sym]
            else:
                q = op.eng.quote(mids[sym], op.inv, mid_history=mid_hist(sym, i))
                if q.quoting and q.ask_size > 0:
                    op.sell_px = float(q.ask)

        # New buys on free symbols with positive AS edge.
        open_count = len(pending) + len(positions)
        for sym in symbols:
            if sym in pending or sym in positions:
                continue
            if open_count >= max_concurrent:
                break
            mid = mids[sym]
            if mid <= 0 or not np.isfinite(mid):
                continue
            notional = float(per_position_usd)
            if cash < notional * 1.01:
                break
            max_inv = max(notional / mid, 1e-12)
            eng = QuoteEngine(cfg.model_copy(update={"max_inventory": max_inv}))
            inv = InventoryState(
                base=0.0,
                quote=0.0,
                target=0.0,
                max_abs=max_inv,
                halt_frac=cfg.inventory_halt_frac,
            )
            q = eng.quote(mid, inv, mid_history=mid_hist(sym, i))
            if (
                (not q.quoting)
                or q.net_edge_bps_if_matched <= 0
                or q.bid_size <= 0
                or q.bid <= 0
            ):
                continue
            size = min(float(q.bid_size), notional / q.bid)
            size = min(size, (cash * 0.999) / (q.bid * (1.0 + cfg.commission_frac)))
            if size <= 0:
                continue
            pending[sym] = PendingBuy(
                symbol=sym,
                px=float(q.bid),
                size=float(size),
                notional=notional,
                placed_i=int(i),
                eng=eng,
                inv=inv,
            )
            n_buy_orders += 1
            open_count += 1

    last_i = int(trade_idx[-1])
    pending_eod = [
        {
            "symbol": pb.symbol,
            "buy_px": pb.px,
            "bars_waiting": int(last_i - pb.placed_i),
            "last_close": float(closes[pb.symbol][last_i]),
        }
        for pb in pending.values()
    ]
    open_eod = [
        {
            "symbol": op.symbol,
            "buy_px": op.buy_px,
            "sell_px": op.sell_px,
            "base": float(op.inv.base),
            "mid": float(closes[op.symbol][last_i]),
            "bars_held": int(last_i - op.entry_i),
        }
        for op in positions.values()
    ]
    inv_m = inv_mark(last_i)
    final_eq = cash + inv_m
    return Result(
        n_bars_trade=int(len(trade_idx)),
        n_buy_orders=int(n_buy_orders),
        n_buy_fills=int(n_buy_fills),
        n_targets=int(n_targets),
        n_pending_eod=len(pending_eod),
        n_open_eod=len(open_eod),
        n_sell_fills=int(n_sell_fills),
        fees_paid=float(fees_total),
        realized_pnl=float(realized_total),
        start_equity=float(initial_equity),
        final_equity=float(final_eq),
        return_pct=float(100.0 * (final_eq / initial_equity - 1.0)),
        cash_end=float(cash),
        inventory_mark_end=float(inv_m),
        avg_bars_to_buy_fill=(
            float(np.mean(buy_wait_bars)) if buy_wait_bars else None
        ),
        avg_bars_held_to_target=(float(np.mean(hold_bars)) if hold_bars else None),
        interpretation_allowed=False,
        notes=(
            f"OHLC touch; ${per_position_usd:.0f}/pos; no buy timeout; "
            f"independent symbols; fee {cfg.commission_pct}%/side."
        ),
        targets_by_symbol=dict(
            sorted(targets_by_symbol.items(), key=lambda kv: -kv[1])
        ),
        targets_sample=target_rows[:20],
        open_eod_sample=open_eod[:20],
        pending_eod_sample=pending_eod[:20],
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config/live_okx_tr_1m_grow150.yaml")
    p.add_argument("--equity", type=float, default=10_000_000.0)
    p.add_argument("--per-position", type=float, default=200.0)
    p.add_argument("--warmup-start", default="2026-08-17")
    p.add_argument("--trade-start", default="2026-09-01")
    p.add_argument("--trade-end", default="2026-09-02")
    p.add_argument("--twelve", action="store_true")
    p.add_argument(
        "--out",
        default="config/ohlc_touch_parallel_200_sep1_2026_results.json",
    )
    args = p.parse_args(argv)

    cfg = load_mm_config(args.config)
    symbols = TWELVE if args.twelve else list(cfg.symbols)
    stamps, panel = load_ohlc_panel(
        symbols,
        timeframe="1m",
        start=args.warmup_start,
        end_exclusive=args.trade_end,
        root=cfg.storage_root,
        exchange_id=cfg.exchange_id,
    )
    res = run_parallel_touch(
        cfg,
        panel,
        stamps,
        trade_start=args.trade_start,
        trade_end_exclusive=args.trade_end,
        initial_equity=float(args.equity),
        per_position_usd=float(args.per_position),
    )
    payload = {
        "interpretation_allowed": False,
        "design": {
            "mode": "ohlc_touch_parallel_no_timeout",
            "start_equity": args.equity,
            "per_position_usd": args.per_position,
            "buy_timeout_bars": None,
            "collision": "none",
            "trade_window": f"{args.trade_start}->{args.trade_end}",
            "warmup_start": args.warmup_start,
            "n_symbols": len(symbols),
            "symbols": symbols,
            "fill_model": "low<=bid / high>=ask; no RNG",
            "params": args.config,
        },
        "result": asdict(res),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    r = payload["result"]
    keys = [
        "n_bars_trade", "n_buy_orders", "n_buy_fills", "n_targets",
        "n_pending_eod", "n_open_eod", "fees_paid", "realized_pnl",
        "start_equity", "final_equity", "return_pct",
        "avg_bars_to_buy_fill", "avg_bars_held_to_target", "targets_by_symbol",
    ]
    print(json.dumps({k: r[k] for k in keys}, indent=2), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
