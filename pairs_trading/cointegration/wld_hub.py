"""WLD hub/placebo screen + residual cross-correlation of WLD-pair candidates."""

from __future__ import annotations

import itertools
import logging
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from pairs_trading.cointegration.eg_dual_null import dual_null_eg_pair
from pairs_trading.cointegration.engine import align_close_prices, evaluate_pair, ols_hedge_ratio
from pairs_trading.cointegration.fdr import apply_benjamini_hochberg
from pairs_trading.cointegration.persist import save_scan_results
from pairs_trading.cointegration.rolling import common_time_bounds, iter_windows
from pairs_trading.cointegration.scanner import load_symbol_frame, resolve_exchange_id
from pairs_trading.data.config import DataConfig, load_config
from pairs_trading.data.storage import ParquetStore

logger = logging.getLogger(__name__)

# Economically distant placebos for a Worldcoin hub check (user-specified core).
WLD_PLACEBOS = (
    "LTC/USDT",
    "ATOM/USDT",
    "NEAR/USDT",
    "DOGE/USDT",
    "XAUT/USDT",
)

WLD_CANDIDATE_PAIRS = (
    ("OKB/USDT", "WLD/USDT"),
    ("ADA/USDT", "WLD/USDT"),
    ("ARB/USDT", "WLD/USDT"),
)


def _iso_to_ms(iso: str) -> int:
    dt = datetime.strptime(str(iso), "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _slice_ohlcv(df: pd.DataFrame, start_ms: int, end_ms: int) -> pd.DataFrame:
    out = df[(df["timestamp"] >= start_ms) & (df["timestamp"] < end_ms)].copy()
    return out.reset_index(drop=True)


def _involves(sym_a: str, sym_b: str, hub: str) -> bool:
    h = hub.upper()
    return sym_a.upper() == h or sym_b.upper() == h


def rolling_hub_placebo_screen(
    config: DataConfig | None = None,
    *,
    hub_symbol: str = "WLD/USDT",
    placebos: tuple[str, ...] | list[str] | None = None,
    exchange_id: str | None = None,
    window_days: int = 180,
    step_days: int = 90,
    n_sims: int = 1500,
    seed: int = 42,
    persist: bool = True,
    dual_null_on: str = "hub_pairs",  # hub_pairs | raw_hits | all
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Rolling EG among {hub}∪placebos; dual-null on hub pairs; hit-rate summary.

    Returns (eg_rolling_df, dual_null_df, summary_dict).
    """
    config = config or load_config()
    eid = resolve_exchange_id(config, exchange_id)
    store = ParquetStore(config.storage.root, eid, config.ohlcv.timeframe)
    placebos = list(placebos or WLD_PLACEBOS)
    universe = [hub_symbol, *placebos]
    alpha = config.cointegration.pvalue_threshold
    min_obs = min(config.cointegration.min_obs, max(60, window_days - 5))

    frames: dict[str, pd.DataFrame] = {}
    for sym in universe:
        try:
            frames[sym] = load_symbol_frame(store, sym)
        except FileNotFoundError:
            logger.warning("Missing %s — drop from WLD hub screen", sym)
    available = sorted(frames.keys())
    if hub_symbol not in frames or len(available) < 3:
        raise FileNotFoundError(f"Need hub + placebos; have {available}")

    t0, t1 = common_time_bounds(frames)
    windows = iter_windows(t0, t1, window_days=window_days, step_days=step_days)
    pairs = list(itertools.combinations(available, 2))

    eg_rows: list[dict] = []
    for w_start, w_end, s_iso, e_iso in windows:
        sliced = {
            s: _slice_ohlcv(df, w_start, w_end)
            for s, df in frames.items()
            if len(_slice_ohlcv(df, w_start, w_end)) >= min_obs
        }
        for y_sym, x_sym in pairs:
            if y_sym not in sliced or x_sym not in sliced:
                continue
            try:
                hit = evaluate_pair(
                    y_sym,
                    x_sym,
                    sliced[y_sym],
                    sliced[x_sym],
                    min_obs=min_obs,
                    price_space=config.cointegration.price_space,
                )
            except ValueError:
                continue
            row = asdict(hit)
            row.update(
                {
                    "pair": f"{y_sym}|{x_sym}",
                    "window_start": s_iso,
                    "window_end": e_iso,
                    "window_days": window_days,
                    "contains_hub": _involves(y_sym, x_sym, hub_symbol),
                    "significant_raw": hit.pvalue < alpha,
                }
            )
            eg_rows.append(row)

    eg_df = pd.DataFrame(eg_rows)
    if eg_df.empty:
        return eg_df, pd.DataFrame(), {}

    # Per-window FDR among this small family (optional diagnostic)
    pieces = []
    for (ws, we), g in eg_df.groupby(["window_start", "window_end"]):
        pieces.append(apply_benjamini_hochberg(g, pvalue_col="pvalue", alpha=alpha))
    eg_df = pd.concat(pieces, ignore_index=True)

    # Manual rates (contains_hub flag is authoritative for this closed universe)
    hub_mask = eg_df["contains_hub"]
    rate_hub = float(eg_df.loc[hub_mask, "significant_raw"].mean()) if hub_mask.any() else float("nan")
    rate_other = (
        float(eg_df.loc[~hub_mask, "significant_raw"].mean()) if (~hub_mask).any() else float("nan")
    )
    summary = {
        "hub_symbol": hub_symbol,
        "n_windows": int(eg_df[["window_start", "window_end"]].drop_duplicates().shape[0]),
        "n_hub_pair_windows": int(hub_mask.sum()),
        "n_other_pair_windows": int((~hub_mask).sum()),
        "n_hub_raw_hits": int(eg_df.loc[hub_mask, "significant_raw"].sum()),
        "n_other_raw_hits": int(eg_df.loc[~hub_mask, "significant_raw"].sum()),
        "hit_rate_hub": rate_hub,
        "hit_rate_other": rate_other,
        "hit_rate_ratio": (rate_hub / rate_other) if rate_other and rate_other > 0 else float("inf"),
    }

    # Dual-null subset
    if dual_null_on == "all":
        dual_src = eg_df
    elif dual_null_on == "raw_hits":
        dual_src = eg_df[eg_df["significant_raw"]]
    else:  # hub_pairs
        dual_src = eg_df[eg_df["contains_hub"]]

    dual_rows: list[dict] = []
    for i, row in dual_src.reset_index(drop=True).iterrows():
        y_sym, x_sym = str(row["y_symbol"]), str(row["x_symbol"])
        w0, w1 = _iso_to_ms(row["window_start"]), _iso_to_ms(row["window_end"])
        y_df = _slice_ohlcv(frames[y_sym], w0, w1)
        x_df = _slice_ohlcv(frames[x_sym], w0, w1)
        try:
            y_log, x_log = align_close_prices(
                y_df,
                x_df,
                min_obs=min_obs,
                price_space=config.cointegration.price_space,
            )
            dual = dual_null_eg_pair(
                y_log.to_numpy(),
                x_log.to_numpy(),
                n_sims=n_sims,
                seed=seed + 17 * (i + 1),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Dual-null failed %s: %s", row["pair"], exc)
            continue
        dual_rows.append(
            {
                "window_start": row["window_start"],
                "window_end": row["window_end"],
                "window_days": window_days,
                "pair": row["pair"],
                "y_symbol": y_sym,
                "x_symbol": x_sym,
                "contains_hub": bool(row["contains_hub"]),
                "pvalue_table": float(row["pvalue"]),
                "significant_raw": bool(row["significant_raw"]),
                **dual,
            }
        )
        logger.info(
            "WLD dual %s %s→%s: table=%.4f rw=%.4f bb=%.4f",
            row["pair"],
            row["window_start"],
            row["window_end"],
            row["pvalue"],
            dual["pvalue_mc"],
            dual["pvalue_bb"],
        )

    dual_df = pd.DataFrame(dual_rows)
    if not dual_df.empty and dual_df["contains_hub"].any():
        hub_dual = dual_df[dual_df["contains_hub"]]
        summary["dual_hub_n"] = int(len(hub_dual))
        summary["dual_hub_raw_hit_rate"] = float(hub_dual["significant_raw"].mean())
        summary["dual_hub_median_p_rw"] = float(hub_dual["pvalue_mc"].median())
        summary["dual_hub_median_p_bb"] = float(hub_dual["pvalue_bb"].median())
        summary["dual_hub_frac_bb_lt_05"] = float((hub_dual["pvalue_bb"] < alpha).mean())

    if persist:
        save_scan_results(
            eg_df,
            root=config.storage.root,
            exchange_id=eid,
            timeframe=config.ohlcv.timeframe,
            mode=config.cointegration.persist_mode,
            meta={"kind": "wld_hub_rolling_eg", **summary},
            basename="wld_hub_rolling_eg",
        )
        if not dual_df.empty:
            save_scan_results(
                dual_df,
                root=config.storage.root,
                exchange_id=eid,
                timeframe=config.ohlcv.timeframe,
                mode=config.cointegration.persist_mode,
                meta={"kind": "wld_hub_dual_null", "n_sims": n_sims, "seed": seed},
                basename="wld_hub_dual_null",
            )
    return eg_df, dual_df, summary


def candidate_spread_residuals(
    config: DataConfig | None = None,
    *,
    pairs: tuple[tuple[str, str], ...] = WLD_CANDIDATE_PAIRS,
    exchange_id: str | None = None,
    window_start: str | None = None,
    window_end: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """OLS spreads for candidate WLD pairs; return aligned residual panel + corr."""
    config = config or load_config()
    eid = resolve_exchange_id(config, exchange_id)
    store = ParquetStore(config.storage.root, eid, config.ohlcv.timeframe)

    pieces: list[pd.DataFrame] = []
    for y_sym, x_sym in pairs:
        y_df = load_symbol_frame(store, y_sym)
        x_df = load_symbol_frame(store, x_sym)
        if window_start and window_end:
            w0, w1 = _iso_to_ms(window_start), _iso_to_ms(window_end)
            y_df, x_df = _slice_ohlcv(y_df, w0, w1), _slice_ohlcv(x_df, w0, w1)
        y, x = align_close_prices(
            y_df,
            x_df,
            min_obs=60,
            price_space=config.cointegration.price_space,
        )
        # Rebuild timestamps via the same inner join as align_close_prices
        left = y_df[["timestamp", "close"]].rename(columns={"close": "_y"})
        right = x_df[["timestamp", "close"]].rename(columns={"close": "_x"})
        merged = left.merge(right, on="timestamp", how="inner").sort_values("timestamp")
        merged = merged.dropna().reset_index(drop=True)
        if len(merged) != len(y):
            merged = merged.iloc[: len(y)].reset_index(drop=True)
        _beta, _a, spread = ols_hedge_ratio(y.reset_index(drop=True), x.reset_index(drop=True))
        name = f"{y_sym.split('/')[0]}-{x_sym.split('/')[0]}"
        pieces.append(
            pd.DataFrame({"timestamp": merged["timestamp"].to_numpy(), name: spread.to_numpy()})
        )

    panel = pieces[0]
    for p in pieces[1:]:
        panel = panel.merge(p, on="timestamp", how="inner")
    panel = panel.sort_values("timestamp").dropna()
    cols = [c for c in panel.columns if c != "timestamp"]
    levels = panel[cols]
    diffs = levels.diff().dropna()
    corr_level = levels.corr()
    corr_diff = diffs.corr()

    corr_long = corr_level.stack().rename("corr_level").reset_index()
    corr_long.columns = ["pair_a", "pair_b", "corr_level"]
    diff_long = corr_diff.stack().rename("corr_diff").reset_index()
    diff_long.columns = ["pair_a", "pair_b", "corr_diff"]
    corr_long = corr_long.merge(diff_long, on=["pair_a", "pair_b"])
    return panel, corr_long


def correlate_wld_candidate_residuals(
    config: DataConfig | None = None,
    *,
    exchange_id: str | None = None,
    persist: bool = True,
) -> dict[str, pd.DataFrame]:
    """Correlate OKB-WLD / ADA-WLD / ARB-WLD spreads on full overlap + key windows."""
    config = config or load_config()
    eid = resolve_exchange_id(config, exchange_id)

    windows = [
        ("full", None, None),
        ("okb_wld_fdr", "2024-03-08", "2024-09-04"),
        ("ada_wld_fdr", "2024-12-03", "2025-06-01"),
        ("arb_wld_fdr", "2024-03-08", "2025-03-08"),
    ]
    frames: dict[str, pd.DataFrame] = {}
    for label, ws, we in windows:
        panel, corr = candidate_spread_residuals(
            config,
            exchange_id=eid,
            window_start=ws,
            window_end=we,
        )
        corr = corr.copy()
        corr["sample"] = label
        frames[label] = corr
        # off-diagonal mean |corr|
        off = corr[corr["pair_a"] != corr["pair_b"]]
        logger.info(
            "Residual corr [%s]: mean |level|=%.3f mean |diff|=%.3f",
            label,
            off["corr_level"].abs().mean(),
            off["corr_diff"].abs().mean(),
        )

    stacked = pd.concat(frames.values(), ignore_index=True)
    if persist:
        save_scan_results(
            stacked,
            root=config.storage.root,
            exchange_id=eid,
            timeframe=config.ohlcv.timeframe,
            mode=config.cointegration.persist_mode,
            meta={"kind": "wld_candidate_residual_corr", "pairs": [f"{a}|{b}" for a, b in WLD_CANDIDATE_PAIRS]},
            basename="wld_candidate_residual_corr",
        )
    return {"corr": stacked, **frames}
