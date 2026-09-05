"""Paper simulator: GBM mid + queue-intensity fills + fees + adverse selection."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from market_making.config import MarketMakingConfig
from market_making.engine import QuoteEngine
from market_making.fees import fee_on_notional
from market_making.inventory import InventoryState


@dataclass
class SimResult:
    n_bars: int
    n_buys: int
    n_sells: int
    fees_paid: float
    realized_pnl: float
    start_equity: float
    final_equity: float
    final_inventory: float
    total_return_pct: float
    net_positive: bool
    avg_half_spread_bps: float
    required_half_spread_bps: float
    commission_pct: float
    commission_bps: float
    interpretation_allowed: bool
    notes: str


def _gbm_path(
    n: int, mid0: float, ann_vol: float, bars_per_year: float, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dt = 1.0 / bars_per_year
    shocks = rng.normal(0.0, ann_vol * np.sqrt(dt), size=n)
    return mid0 * np.exp(np.cumsum(shocks))


def _fill_prob(mid: float, quote_px: float, half_bps: float, base_p: float) -> float:
    """Maker-queue intensity: higher when mid is near the quote."""
    half = max(half_bps / 1e4, 1e-8)
    gap = abs(mid - quote_px) / mid
    return float(base_p * np.exp(-gap / half))


def run_mid_path_sim(
    cfg: MarketMakingConfig,
    mids: np.ndarray,
    *,
    seed: int | None = None,
    timestamps: np.ndarray | None = None,
    notes: str | None = None,
) -> tuple[SimResult, pd.DataFrame]:
    """Core MM loop over an arbitrary mid path (GBM or historical)."""
    mids = np.asarray(mids, dtype=float)
    if len(mids) < 2:
        raise ValueError("Need at least 2 mid bars")
    sc = cfg.sim
    seed = sc.seed if seed is None else seed
    eng = QuoteEngine(cfg)
    mid0 = float(mids[0])
    inv = InventoryState(
        base=0.0,
        quote=cfg.max_inventory * mid0 * 2.0,
        target=cfg.target_inventory,
        max_abs=cfg.max_inventory,
        halt_frac=cfg.inventory_halt_frac,
    )
    start_eq = inv.mark_to_market(mid0)
    rng = np.random.default_rng(seed + 17)
    rows: list[dict[str, Any]] = []
    halfs: list[float] = []

    for i, mid in enumerate(mids):
        mid_f = float(mid)
        q = eng.quote(mid_f, inv, mid_history=mids[: i + 1])
        if q.quoting:
            halfs.append(q.half_spread_bps)

        buy_fill = False
        sell_fill = False
        if q.quoting and q.bid_size > 0:
            p_bid = _fill_prob(mid_f, q.bid, q.half_spread_bps, sc.fill_prob)
            if rng.random() < p_bid:
                px = q.bid * (1.0 + sc.adverse_frac * (q.half_spread_bps / 1e4))
                fee = fee_on_notional(px * q.bid_size, cfg)
                inv.apply_buy(price=px, size=q.bid_size, fee=fee, bar=i)
                buy_fill = True
        if q.quoting and q.ask_size > 0:
            p_ask = _fill_prob(mid_f, q.ask, q.half_spread_bps, sc.fill_prob)
            if rng.random() < p_ask:
                px = q.ask * (1.0 - sc.adverse_frac * (q.half_spread_bps / 1e4))
                fee = fee_on_notional(px * q.ask_size, cfg)
                inv.apply_sell(price=px, size=q.ask_size, fee=fee, bar=i)
                sell_fill = True

        row: dict[str, Any] = {
            "bar": i,
            "mid": mid_f,
            "bid": q.bid if q.quoting else np.nan,
            "ask": q.ask if q.quoting else np.nan,
            "half_spread_bps": q.half_spread_bps if q.quoting else np.nan,
            "net_edge_bps": q.net_edge_bps_if_matched if q.quoting else np.nan,
            "quoting": q.quoting,
            "reason": q.reason,
            "inventory": inv.base,
            "equity": inv.mark_to_market(mid_f),
            "buy_fill": buy_fill,
            "sell_fill": sell_fill,
            "fees_cum": inv.fees_paid,
            "realized_pnl": inv.realized_pnl,
        }
        if timestamps is not None:
            row["timestamp"] = timestamps[i]
        rows.append(row)

    final_eq = inv.mark_to_market(float(mids[-1]))
    ret = 100.0 * (final_eq - start_eq) / start_eq if start_eq else 0.0
    trades = inv.n_buys + inv.n_sells
    default_notes = (
        f"Queue-fill MM. Gate half-spread>="
        f"{cfg.required_half_spread_bps:.1f}bps "
        f"(fee {cfg.commission_bps:.1f}bps + buffer {cfg.edge_buffer_bps:.1f}bps). "
        "Not a live-edge claim."
    )
    result = SimResult(
        n_bars=int(len(mids)),
        n_buys=inv.n_buys,
        n_sells=inv.n_sells,
        fees_paid=float(inv.fees_paid),
        realized_pnl=float(inv.realized_pnl),
        start_equity=float(start_eq),
        final_equity=float(final_eq),
        final_inventory=float(inv.base),
        total_return_pct=float(ret),
        net_positive=bool(final_eq > start_eq and trades > 0),
        avg_half_spread_bps=float(np.mean(halfs)) if halfs else 0.0,
        required_half_spread_bps=float(cfg.required_half_spread_bps),
        commission_pct=float(cfg.commission_pct),
        commission_bps=float(cfg.commission_bps),
        interpretation_allowed=False,
        notes=notes or default_notes,
    )
    return result, pd.DataFrame(rows)


def run_paper_sim(cfg: MarketMakingConfig) -> tuple[SimResult, pd.DataFrame]:
    sc = cfg.sim
    mids = _gbm_path(sc.n_bars, sc.mid0, sc.ann_vol, sc.bars_per_year, sc.seed)
    return run_mid_path_sim(
        cfg,
        mids,
        seed=sc.seed,
        notes=(
            f"Paper GBM+queue fills. Gate half-spread>="
            f"{cfg.required_half_spread_bps:.1f}bps "
            f"(fee {cfg.commission_bps:.1f}bps + buffer {cfg.edge_buffer_bps:.1f}bps). "
            "Not a live-edge claim."
        ),
    )


def load_okx_mids(
    symbol: str,
    *,
    timeframe: str = "4h",
    start: str,
    end_exclusive: str,
    root: str | Path = "data",
    exchange_id: str = "okx",
) -> tuple[np.ndarray, np.ndarray]:
    """Load OKX OHLCV closes as mids for [start, end_exclusive)."""
    base = symbol.replace("/", "_")
    path = Path(root) / exchange_id / timeframe / f"{base}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    mask = (ts >= pd.Timestamp(start, tz="UTC")) & (
        ts < pd.Timestamp(end_exclusive, tz="UTC")
    )
    sl = df.loc[mask].sort_values("timestamp")
    if len(sl) < 2:
        raise ValueError(f"Insufficient bars for {symbol} {start}→{end_exclusive}: {len(sl)}")
    mids = sl["close"].to_numpy(dtype=float)
    stamps = ts.loc[mask].sort_values().to_numpy()
    return mids, stamps


def run_month_sim(
    cfg: MarketMakingConfig,
    *,
    year: int,
    month: int = 1,
    symbol: str = "BTC/USDT",
    timeframe: str = "4h",
    seed: int | None = None,
) -> tuple[SimResult, pd.DataFrame]:
    """Run MM on historical OKX bars for one calendar month."""
    start = f"{year:04d}-{month:02d}-01"
    if month == 12:
        end = f"{year + 1:04d}-01-01"
    else:
        end = f"{year:04d}-{month + 1:02d}-01"
    mids, stamps = load_okx_mids(
        symbol,
        timeframe=timeframe,
        start=start,
        end_exclusive=end,
        root=cfg.storage_root,
        exchange_id=cfg.exchange_id,
    )
    # Scale max inventory to ~$10k notional equivalent for high-priced coins.
    # Keep quote cash = 2× max base × mid0.
    notes = (
        f"Historical {symbol} {timeframe} {start}→{end} on {cfg.exchange_id}. "
        f"n_bars={len(mids)}. Queue fills + fee {cfg.commission_pct}%/side. "
        "Not a live-edge claim."
    )
    return run_mid_path_sim(cfg, mids, seed=seed, timestamps=stamps, notes=notes)


def save_sim_artifacts(
    result: SimResult,
    frame: pd.DataFrame,
    *,
    root: str | Path | None = None,
    exchange_id: str = "okx",
    basename: str = "paper_sim_latest",
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
