"""$50k-scale capacity-aware MM research profile (pre-2026).

Design (liquidity-constrained)
------------------------------
- Universe: BTC / ETH / SOL only (2025 1h: only these absorb ~$15k–$50k
  clips at ≤1% of thin-hour USD volume; alts do not).
- Wallet: $50,000.
- Max concurrent slots: 2 × $25,000 (never a single $50k dump into one alt).
- Smaller ``size_frac_of_max`` so each touch is a fraction of the slot.
- Wider fee/adverse buffers than the $3.5k retail profile.

Calibration uses 2025 only (pre-2026). Not a live-edge claim.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from market_making.config import MarketMakingConfig, load_mm_config
from market_making.multi_portfolio import (
    run_multi_portfolio_range,
    save_multi_portfolio_artifacts,
)
from market_making.simulator import run_month_sim


LARGE_UNIVERSE = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]


def base_50k_config(cfg: MarketMakingConfig | None = None) -> MarketMakingConfig:
    """Start from current MM yaml, then apply $50k capacity knobs."""
    base = cfg or load_mm_config()
    sim = base.sim.model_copy(update={"fill_prob": 0.20, "adverse_frac": 0.60})
    return base.model_copy(
        update={
            "symbols": list(LARGE_UNIVERSE),
            "portfolio_initial_equity": 50_000.0,
            "size_frac_of_max": 0.05,  # smaller clips vs retail 0.10
            "min_half_spread_bps": 14.0,
            "edge_buffer_bps": 6.0,
            "adverse_vol_mult": 0.55,
            "inventory_halt_frac": 0.75,
            "vol_lookback": 192,
            "ewma_halflife": 96.0,
            "liq_bps": 8.0,
            "sim": sim,
            "interpretation_allowed": False,
        }
    )


def large_param_grid() -> list[dict[str, float]]:
    """Compact AS grid for large-book research (1h units)."""
    gammas = [0.12, 0.18, 0.25]
    kappas = [0.6, 0.8, 1.2]
    taus = [24.0, 48.0, 72.0]
    etas = [2.0, 3.0]
    out: list[dict[str, float]] = []
    for g, k, t, e in itertools.product(gammas, kappas, taus, etas):
        out.append(
            {
                "gamma": float(g),
                "kappa": float(k),
                "tau_bars": float(t),
                "size_eta": float(e),
            }
        )
    return out


def _apply(cfg: MarketMakingConfig, params: dict[str, float]) -> MarketMakingConfig:
    return cfg.model_copy(update=params)


def _eval_symbol_months(
    cfg: MarketMakingConfig,
    months: list[tuple[int, int]],
    *,
    timeframe: str,
    seed: int,
) -> dict[str, float]:
    rets: list[float] = []
    for y, m in months:
        for sym in cfg.symbols:
            r, _ = run_month_sim(
                cfg, year=y, month=m, symbol=sym, timeframe=timeframe, seed=seed
            )
            rets.append(float(r.total_return_pct))
    arr = np.asarray(rets, dtype=float)
    return {
        "n": float(len(arr)),
        "pct_pos": float(100.0 * np.mean(arr > 0)) if len(arr) else 0.0,
        "mean_ret": float(arr.mean()) if len(arr) else 0.0,
        "min_ret": float(arr.min()) if len(arr) else 0.0,
    }


@dataclass
class LargeBookCalibResult:
    label: str
    universe: list[str]
    wallet_usd: float
    max_slots: int
    slot_notional_usd: float
    selected_params: dict[str, float]
    is_months: list[str]
    oos_months: list[str]
    is_stats: dict[str, float]
    oos_stats: dict[str, float]
    top_candidates: list[dict[str, Any]]
    portfolio_is: dict[str, Any]
    portfolio_oos: dict[str, Any]
    interpretation_allowed: bool
    method: str
    created_at_utc: str


def calibrate_large_book_2025(
    cfg: MarketMakingConfig | None = None,
    *,
    timeframe: str = "1h",
    seed: int = 42,
    max_slots: int = 2,
    slot_notional_usd: float = 25_000.0,
    wallet_usd: float = 50_000.0,
) -> LargeBookCalibResult:
    """IS = 2025 H1, OOS = 2025 H2. Pre-2026 research only."""
    cfg0 = base_50k_config(cfg)
    is_months = [(2025, m) for m in range(1, 7)]
    oos_months = [(2025, m) for m in range(7, 13)]
    grid = large_param_grid()

    ranked: list[tuple[tuple[float, float, float], dict[str, float], dict[str, float]]] = []
    for i, params in enumerate(grid):
        trial = _apply(cfg0, params)
        stats = _eval_symbol_months(trial, is_months, timeframe=timeframe, seed=seed)
        score = (stats["pct_pos"], stats["mean_ret"], stats["min_ret"])
        ranked.append((score, params, stats))
        if (i + 1) % 12 == 0:
            print(f"large50k calib {i+1}/{len(grid)}", flush=True)

    ranked.sort(key=lambda x: x[0], reverse=True)
    best_params, best_is = ranked[0][1], ranked[0][2]
    best_cfg = _apply(cfg0, best_params)
    oos_stats = _eval_symbol_months(
        best_cfg, oos_months, timeframe=timeframe, seed=seed
    )

    # Portfolio compounds on IS / OOS windows
    def _compound(months: list[tuple[int, int]]) -> dict[str, Any]:
        start = f"{months[0][0]:04d}-{months[0][1]:02d}-01"
        y2, m2 = months[-1]
        end = f"{y2+1:04d}-01-01" if m2 == 12 else f"{y2:04d}-{m2+1:02d}-01"
        res, _ = run_multi_portfolio_range(
            best_cfg,
            start=start,
            end_exclusive=end,
            timeframe=timeframe,
            seed=seed,
            initial_equity=wallet_usd,
            max_slots=max_slots,
            slot_notional_usd=slot_notional_usd,
            symbols=list(LARGE_UNIVERSE),
        )
        return {
            "start": start,
            "end_exclusive": end,
            "start_equity": res.start_equity,
            "final_equity": res.final_equity,
            "total_return_pct": res.total_return_pct,
            "n_entries": res.n_entries,
            "n_round_trips": res.n_round_trips,
            "fees_paid": res.fees_paid,
        }

    print("portfolio IS 2025H1...", flush=True)
    port_is = _compound(is_months)
    print("portfolio OOS 2025H2...", flush=True)
    port_oos = _compound(oos_months)

    top = [
        {"params": p, "is_stats": s, "score": list(sc)}
        for sc, p, s in ranked[:8]
    ]
    return LargeBookCalibResult(
        label="large_book_50k_v1_pre2026",
        universe=list(LARGE_UNIVERSE),
        wallet_usd=float(wallet_usd),
        max_slots=int(max_slots),
        slot_notional_usd=float(slot_notional_usd),
        selected_params={**best_params, "size_frac_of_max": 0.05, "fill_prob": 0.20, "adverse_frac": 0.60},
        is_months=[f"{y:04d}-{m:02d}" for y, m in is_months],
        oos_months=[f"{y:04d}-{m:02d}" for y, m in oos_months],
        is_stats=best_is,
        oos_stats=oos_stats,
        top_candidates=top,
        portfolio_is=port_is,
        portfolio_oos=port_oos,
        interpretation_allowed=False,
        method=(
            "Pre-2026 large-book research. Universe BTC/ETH/SOL. "
            "IS=2025H1 symbol×month grid; OOS=2025H2. "
            "Wallet $50k, 2×$25k slots, size_frac=0.05, fill=0.20. Not live edge."
        ),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def save_large_book_calib(
    result: LargeBookCalibResult,
    *,
    root: str | Path = "data",
    exchange_id: str = "okx",
    basename: str = "large_book_50k_v1_pre2026",
) -> dict[str, str]:
    out = Path(root) / exchange_id / "market_making"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{basename}.json"
    payload = asdict(result)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    cfg_path = Path("config") / f"{basename}.yaml"
    # Write a dedicated profile yaml (does not overwrite retail market_making.yaml)
    retail = load_mm_config()
    mm = {
        "symbols": result.universe,
        "commission_pct": retail.commission_pct,
        "edge_buffer_bps": 6.0,
        "min_half_spread_bps": 14.0,
        "inventory_skew_bps": retail.inventory_skew_bps,
        "target_inventory": 0.0,
        "max_inventory": 1.0,
        "size_frac_of_max": 0.05,
        "inventory_halt_frac": 0.75,
        "vol_lookback": 192,
        "vol_spread_mult": 0.25,
        "gamma": result.selected_params["gamma"],
        "kappa": result.selected_params["kappa"],
        "tau_bars": result.selected_params["tau_bars"],
        "ewma_halflife": 96.0,
        "size_eta": result.selected_params["size_eta"],
        "adverse_vol_mult": 0.55,
        "liq_bps": 8.0,
        "use_as_quotes": True,
        "portfolio_initial_equity": result.wallet_usd,
        "portfolio_position_frac": 1.0,
        "realtime_poll_sec": 1.0,
        "portfolio_flat_eps": 1.0e-8,
        "interpretation_allowed": False,
        "sim": {
            "n_bars": 2000,
            "seed": 42,
            "mid0": 100.0,
            "ann_vol": 0.55,
            "fill_prob": 0.20,
            "adverse_frac": 0.60,
            "bars_per_year": 8760,
        },
    }
    doc = {
        "exchange": {"id": "okx", "fallbacks": ["bybit", "binance"], "quote": "USDT", "market_type": "spot"},
        "storage": {"root": "data"},
        "market_making": mm,
        "large_book": {
            "max_slots": result.max_slots,
            "slot_notional_usd": result.slot_notional_usd,
            "label": result.label,
        },
    }
    cfg_path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")

    art = Path("/opt/cursor/artifacts")
    art.mkdir(parents=True, exist_ok=True)
    art_path = art / f"{basename}.json"
    art_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return {"summary": str(path), "config": str(cfg_path), "artifact": str(art_path)}
