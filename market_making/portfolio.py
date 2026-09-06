"""Single-wallet portfolio router across the MM universe.

Rules
-----
1. One shared USDT wallet (no per-coin capital silos).
2. While a position is open on symbol S, capital is locked: other symbols
   cannot open until inventory on S is flat again (round-trip complete).
3. When flat and several symbols signal on the same bar, enter only the
   highest-advantage name (score = net_edge_bps_if_matched).

Spot long-only: entries open via bid fills; exits via ask fills.
Paper fill model matches the single-symbol simulator. Not a live-edge claim.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from market_making.config import MarketMakingConfig
from market_making.engine import Quote, QuoteEngine
from market_making.fees import fee_on_notional
from market_making.inventory import InventoryState
from market_making.simulator import _fill_prob, load_okx_mids


@dataclass
class PortfolioResult:
    n_bars: int
    n_entries: int
    n_round_trips: int
    n_buys: int
    n_sells: int
    n_skipped_busy: int
    n_multi_signal_choices: int
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


def _month_bounds(year: int, month: int) -> tuple[str, str]:
    start = f"{year:04d}-{month:02d}-01"
    end = f"{year + 1:04d}-01-01" if month == 12 else f"{year:04d}-{month + 1:02d}-01"
    return start, end


def load_aligned_panel(
    symbols: list[str],
    *,
    timeframe: str,
    start: str,
    end_exclusive: str,
    root: str | Path,
    exchange_id: str,
) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray]]:
    """Inner-join OKX mids on timestamp so all symbols share the same bars."""
    frames: dict[str, pd.Series] = {}
    for sym in symbols:
        mids, stamps = load_okx_mids(
            sym,
            timeframe=timeframe,
            start=start,
            end_exclusive=end_exclusive,
            root=root,
            exchange_id=exchange_id,
        )
        frames[sym] = pd.Series(mids, index=pd.DatetimeIndex(stamps), name=sym)
    panel = pd.concat(frames, axis=1, join="inner").sort_index().dropna(how="any")
    if len(panel) < 2:
        raise ValueError(
            f"Aligned panel too short for {symbols} {start}\u2192{end_exclusive}: {len(panel)}"
        )
    series = {c: panel[c].to_numpy(dtype=float) for c in panel.columns}
    return pd.DatetimeIndex(panel.index), series


def _advantage_score(q: Quote) -> float:
    """Higher = more advantageous. Uses fee-aware matched net edge."""
    if not q.quoting:
        return float("-inf")
    return float(q.net_edge_bps_if_matched)


def run_portfolio_panel(
    cfg: MarketMakingConfig,
    panel: dict[str, np.ndarray],
    *,
    timestamps: pd.DatetimeIndex | np.ndarray | None = None,
    seed: int = 42,
    initial_equity: float | None = None,
    notes: str | None = None,
) -> tuple[PortfolioResult, pd.DataFrame]:
    """Run single-wallet MM over an aligned mid panel."""
    symbols = list(panel.keys())
    if not symbols:
        raise ValueError("Empty panel")
    n = len(next(iter(panel.values())))
    if any(len(v) != n for v in panel.values()):
        raise ValueError("Panel series length mismatch")
    if n < 2:
        raise ValueError("Need at least 2 bars")

    equity0 = float(
        initial_equity if initial_equity is not None else cfg.portfolio_initial_equity
    )
    if equity0 <= 0:
        raise ValueError("initial_equity must be > 0")

    pos_frac = float(np.clip(cfg.portfolio_position_frac, 0.05, 1.0))
    flat_eps = float(cfg.portfolio_flat_eps)
    sc = cfg.sim
    rng = np.random.default_rng(seed + 91)

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
    hist: dict[str, list[float]] = {s: [] for s in symbols}

    n_entries = 0
    n_round_trips = 0
    n_skipped_busy = 0
    n_multi = 0
    active_bars: dict[str, int] = {s: 0 for s in symbols}
    fees_total = 0.0
    realized_total = 0.0
    n_buys = 0
    n_sells = 0
    rows: list[dict[str, Any]] = []

    def mark_equity(mids_now: dict[str, float]) -> float:
        if active is None or abs(inv.base) <= flat_eps:
            return float(cash)
        return float(cash + inv.base * mids_now[active])

    def arm_symbol(sym: str, mid: float) -> None:
        nonlocal inv, eng
        notional = max(cash * pos_frac, 1e-9)
        max_inv = notional / mid
        inv = InventoryState(
            base=0.0,
            quote=0.0,
            target=0.0,
            max_abs=max_inv,
            halt_frac=cfg.inventory_halt_frac,
        )
        eng = QuoteEngine(cfg.model_copy(update={"max_inventory": max_inv}))

    for i in range(n):
        mids_now = {s: float(panel[s][i]) for s in symbols}
        for s in symbols:
            hist[s].append(mids_now[s])
        ts_val = timestamps[i] if timestamps is not None else None

        chosen: str | None = None
        action = "idle"
        n_cand = 0
        busy_signals = 0
        buy_fill = False
        sell_fill = False
        buy_size = 0.0
        sell_size = 0.0
        half_bps = np.nan
        net_edge = np.nan
        adv_score = np.nan

        if active is not None:
            # Capital locked on `active`.
            active_bars[active] += 1
            for s in symbols:
                if s == active:
                    continue
                mid_s = mids_now[s]
                if mid_s <= 0 or not np.isfinite(mid_s):
                    continue
                max_inv = max((cash * pos_frac) / mid_s, 1e-12)
                probe_eng = QuoteEngine(cfg.model_copy(update={"max_inventory": max_inv}))
                probe_inv = InventoryState(
                    base=0.0,
                    quote=0.0,
                    target=0.0,
                    max_abs=max_inv,
                    halt_frac=cfg.inventory_halt_frac,
                )
                q_s = probe_eng.quote(
                    mid_s, probe_inv, mid_history=np.asarray(hist[s], dtype=float)
                )
                if not q_s.quoting or q_s.net_edge_bps_if_matched <= 0:
                    continue
                p_bid = _fill_prob(mid_s, q_s.bid, q_s.half_spread_bps, sc.fill_prob)
                if rng.random() < p_bid:
                    busy_signals += 1
            n_skipped_busy += busy_signals

            mid = mids_now[active]
            q = eng.quote(mid, inv, mid_history=np.asarray(hist[active], dtype=float))
            if q.quoting:
                half_bps = q.half_spread_bps
                net_edge = q.net_edge_bps_if_matched
                adv_score = _advantage_score(q)

                if q.bid_size > 0 and rng.random() < _fill_prob(
                    mid, q.bid, q.half_spread_bps, sc.fill_prob
                ):
                    px = q.bid * (1.0 + sc.adverse_frac * (q.half_spread_bps / 1e4))
                    fee = fee_on_notional(px * q.bid_size, cfg)
                    cost = px * q.bid_size + fee
                    if cost <= cash + 1e-9:
                        cash -= cost
                        prev = inv.realized_pnl
                        inv.apply_buy(price=px, size=q.bid_size, fee=0.0, bar=i)
                        fees_total += fee
                        realized_total += inv.realized_pnl - prev
                        n_buys += 1
                        buy_fill = True
                        buy_size = float(q.bid_size)

                if q.ask_size > 0 and rng.random() < _fill_prob(
                    mid, q.ask, q.half_spread_bps, sc.fill_prob
                ):
                    sell_sz = min(q.ask_size, max(inv.base, 0.0))
                    if sell_sz > flat_eps:
                        px = q.ask * (1.0 - sc.adverse_frac * (q.half_spread_bps / 1e4))
                        fee = fee_on_notional(px * sell_sz, cfg)
                        cash += px * sell_sz - fee
                        prev = inv.realized_pnl
                        inv.apply_sell(price=px, size=sell_sz, fee=0.0, bar=i)
                        fees_total += fee
                        realized_total += inv.realized_pnl - prev
                        n_sells += 1
                        sell_fill = True
                        sell_size = float(sell_sz)

            if abs(inv.base) <= flat_eps:
                if abs(inv.base) > 0:
                    cash += inv.base * mid
                    inv.base = 0.0
                n_round_trips += 1
                chosen = active
                active = None
                action = "close"
            else:
                chosen = active
                action = "manage"

        else:
            # Flat wallet: choose best fillable entry.
            candidates: list[tuple[str, float, Quote, float]] = []
            for s in symbols:
                mid = mids_now[s]
                if mid <= 0 or not np.isfinite(mid):
                    continue
                max_inv = max((cash * pos_frac) / mid, 1e-12)
                local_eng = QuoteEngine(cfg.model_copy(update={"max_inventory": max_inv}))
                flat_inv = InventoryState(
                    base=0.0,
                    quote=0.0,
                    target=0.0,
                    max_abs=max_inv,
                    halt_frac=cfg.inventory_halt_frac,
                )
                q = local_eng.quote(
                    mid, flat_inv, mid_history=np.asarray(hist[s], dtype=float)
                )
                if not q.quoting or q.net_edge_bps_if_matched <= 0 or q.bid_size <= 0:
                    continue
                if rng.random() < _fill_prob(mid, q.bid, q.half_spread_bps, sc.fill_prob):
                    candidates.append((s, _advantage_score(q), q, max_inv))

            n_cand = len(candidates)
            if n_cand > 1:
                n_multi += 1

            if candidates:
                candidates.sort(key=lambda t: t[1], reverse=True)
                sym, score, _q0, max_inv = candidates[0]
                chosen = sym
                mid = mids_now[sym]
                arm_symbol(sym, mid)
                active = sym
                n_entries += 1
                action = "enter"
                active_bars[sym] += 1
                adv_score = score

                q = eng.quote(mid, inv, mid_history=np.asarray(hist[sym], dtype=float))
                if q.quoting:
                    half_bps = q.half_spread_bps
                    net_edge = q.net_edge_bps_if_matched
                    if q.bid_size > 0:
                        px = q.bid * (1.0 + sc.adverse_frac * (q.half_spread_bps / 1e4))
                        fee = fee_on_notional(px * q.bid_size, cfg)
                        cost = px * q.bid_size + fee
                        if cost <= cash + 1e-9:
                            cash -= cost
                            prev = inv.realized_pnl
                            inv.apply_buy(price=px, size=q.bid_size, fee=0.0, bar=i)
                            fees_total += fee
                            realized_total += inv.realized_pnl - prev
                            n_buys += 1
                            buy_fill = True
                            buy_size = float(q.bid_size)
            else:
                action = "flat_wait"

        rows.append(
            {
                "bar": i,
                "timestamp": ts_val,
                "active_symbol": active,
                "action": action,
                "chosen_symbol": chosen,
                "n_candidates": n_cand,
                "skipped_busy_signals": busy_signals,
                "cash": cash,
                "inventory": inv.base,
                "equity": mark_equity(mids_now),
                "buy_fill": buy_fill,
                "sell_fill": sell_fill,
                "buy_size": buy_size,
                "sell_size": sell_size,
                "half_spread_bps": half_bps,
                "net_edge_bps": net_edge,
                "advantage_score": adv_score,
                "fees_cum": fees_total,
                "realized_pnl": realized_total,
            }
        )

    if active is not None and abs(inv.base) > flat_eps:
        mid = float(panel[active][-1])
        cash += inv.base * mid
        inv.base = 0.0
        active = None

    final_eq = float(cash)
    ret = 100.0 * (final_eq - equity0) / equity0 if equity0 else 0.0
    default_notes = (
        f"Single-wallet portfolio MM over {symbols}. "
        "Lock-until-flat; multi-signal \u2192 max net_edge_bps. "
        f"Fee {cfg.commission_pct}%/side. Not a live-edge claim."
    )
    result = PortfolioResult(
        n_bars=int(n),
        n_entries=int(n_entries),
        n_round_trips=int(n_round_trips),
        n_buys=int(n_buys),
        n_sells=int(n_sells),
        n_skipped_busy=int(n_skipped_busy),
        n_multi_signal_choices=int(n_multi),
        fees_paid=float(fees_total),
        realized_pnl=float(realized_total),
        start_equity=float(equity0),
        final_equity=float(final_eq),
        total_return_pct=float(ret),
        net_positive=bool(final_eq > equity0 and n_entries > 0),
        active_symbol_bars=active_bars,
        symbols=symbols,
        commission_pct=float(cfg.commission_pct),
        interpretation_allowed=False,
        notes=notes or default_notes,
    )
    return result, pd.DataFrame(rows)


def run_portfolio_month(
    cfg: MarketMakingConfig,
    *,
    year: int,
    month: int,
    timeframe: str = "4h",
    seed: int = 42,
    initial_equity: float | None = None,
    symbols: list[str] | None = None,
) -> tuple[PortfolioResult, pd.DataFrame]:
    syms = list(symbols or cfg.symbols)
    start, end = _month_bounds(year, month)
    stamps, panel = load_aligned_panel(
        syms,
        timeframe=timeframe,
        start=start,
        end_exclusive=end,
        root=cfg.storage_root,
        exchange_id=cfg.exchange_id,
    )
    notes = (
        f"Unified wallet {year:04d}-{month:02d} {timeframe} on {cfg.exchange_id}. "
        f"symbols={syms}. n_bars={len(stamps)}. "
        f"Lock-until-flat + best net_edge pick. Fee {cfg.commission_pct}%/side. "
        "Not a live-edge claim."
    )
    return run_portfolio_panel(
        cfg,
        panel,
        timestamps=stamps,
        seed=seed,
        initial_equity=initial_equity,
        notes=notes,
    )


def save_portfolio_artifacts(
    result: PortfolioResult,
    frame: pd.DataFrame,
    *,
    root: str | Path | None = None,
    exchange_id: str = "okx",
    basename: str = "portfolio_latest",
) -> dict[str, str]:
    out = Path(root or "data") / exchange_id / "market_making"
    out.mkdir(parents=True, exist_ok=True)
    summary = out / f"{basename}.json"
    parquet = out / f"{basename}.parquet"
    csv = out / f"{basename}.csv"
    summary.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    frame.to_parquet(parquet, index=False)
    frame.to_csv(csv, index=False)
    return {"summary": str(summary), "parquet": str(parquet), "csv": str(csv)}
