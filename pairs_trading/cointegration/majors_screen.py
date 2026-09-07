"""Majors quartet screen: BTC/ETH/LTC/XRP — full dual-null + BY + hub + residual chain."""

from __future__ import annotations

import itertools
import logging
from dataclasses import asdict

import pandas as pd

from pairs_trading.cointegration.eg_dual_null import dual_null_eg_pair
from pairs_trading.cointegration.engine import align_close_prices, evaluate_pair
from pairs_trading.cointegration.fdr import (
    apply_benjamini_hochberg,
    apply_benjamini_yekutieli,
    hub_involvement_summary,
)
from pairs_trading.cointegration.persist import save_scan_results
from pairs_trading.cointegration.rolling import common_time_bounds, iter_windows
from pairs_trading.cointegration.scanner import load_symbol_frame, resolve_exchange_id
from pairs_trading.cointegration.wld_hub import (
    _iso_to_ms,
    _slice_ohlcv,
    candidate_spread_residuals,
)
from pairs_trading.data.config import DataConfig, load_config
from pairs_trading.data.storage import ParquetStore

logger = logging.getLogger(__name__)

MAJORS = ("BTC/USDT", "ETH/USDT", "LTC/USDT", "XRP/USDT")
MAJOR_PAIRS = tuple(itertools.combinations(MAJORS, 2))


def _pair_label(y: str, x: str) -> str:
    a, b = sorted([y.split("/")[0], x.split("/")[0]])
    return f"{a}-{b}"


def _run_dual_on_slice(
    y_sym: str,
    x_sym: str,
    y_df: pd.DataFrame,
    x_df: pd.DataFrame,
    *,
    min_obs: int,
    price_space: str,
    n_sims: int,
    seed: int,
) -> dict:
    hit = evaluate_pair(
        y_sym,
        x_sym,
        y_df,
        x_df,
        min_obs=min_obs,
        price_space=price_space,  # type: ignore[arg-type]
    )
    y_log, x_log = align_close_prices(
        y_df, x_df, min_obs=min_obs, price_space=price_space  # type: ignore[arg-type]
    )
    dual = dual_null_eg_pair(
        y_log.to_numpy(),
        x_log.to_numpy(),
        n_sims=n_sims,
        seed=seed,
    )
    return {
        **asdict(hit),
        "pair": f"{y_sym}|{x_sym}",
        "pair_label": _pair_label(y_sym, x_sym),
        "pvalue_table": hit.pvalue,
        **dual,
    }


def majors_full_sample_screen(
    config: DataConfig | None = None,
    *,
    exchange_id: str | None = None,
    n_sims: int = 2000,
    seed: int = 42,
    persist: bool = True,
    window_start: str | None = "2022-01-01",
    window_end: str | None = None,
) -> pd.DataFrame:
    """Full-sample (optionally clipped) EG + dual-null + BH/BY on 6 major pairs."""
    config = config or load_config()
    eid = resolve_exchange_id(config, exchange_id)
    store = ParquetStore(config.storage.root, eid, config.ohlcv.timeframe)
    alpha = config.cointegration.pvalue_threshold
    min_obs = config.cointegration.min_obs

    frames = {s: load_symbol_frame(store, s) for s in MAJORS}
    if window_start or window_end:
        w0 = _iso_to_ms(window_start) if window_start else 0
        w1 = (
            _iso_to_ms(window_end)
            if window_end
            else int(pd.Timestamp.utcnow().timestamp() * 1000)
        )
        frames = {s: _slice_ohlcv(df, w0, w1) for s, df in frames.items()}

    rows: list[dict] = []
    for i, (y_sym, x_sym) in enumerate(MAJOR_PAIRS):
        try:
            row = _run_dual_on_slice(
                y_sym,
                x_sym,
                frames[y_sym],
                frames[x_sym],
                min_obs=min_obs,
                price_space=config.cointegration.price_space,
                n_sims=n_sims,
                seed=seed + 23 * (i + 1),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Full majors fail %s|%s: %s", y_sym, x_sym, exc)
            continue
        row["sample"] = "full"
        row["window_start"] = window_start or "data_start"
        row["window_end"] = window_end or "data_end"
        rows.append(row)
        logger.info(
            "Majors full %s: table=%.4f rw=%.4f bb=%.4f n=%d",
            row["pair_label"],
            row["pvalue_table"],
            row["pvalue_mc"],
            row["pvalue_bb"],
            row["n_obs"],
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = apply_benjamini_hochberg(df, pvalue_col="pvalue_bb", alpha=alpha)
    df = df.rename(
        columns={
            "pvalue_fdr": "pvalue_bb_bh",
            "significant_fdr": "significant_bb_bh",
        }
    )
    by = apply_benjamini_yekutieli(df, pvalue_col="pvalue_bb", alpha=alpha)
    df["pvalue_bb_by"] = by["pvalue_by"]
    df["significant_bb_by"] = by["significant_by"]
    df["significant_raw"] = df["pvalue_table"] < alpha

    if persist:
        save_scan_results(
            df,
            root=config.storage.root,
            exchange_id=eid,
            timeframe=config.ohlcv.timeframe,
            mode=config.cointegration.persist_mode,
            meta={
                "kind": "majors_full_dual",
                "n_sims": n_sims,
                "n_pairs": len(df),
                "n_bb_bh": int(df["significant_bb_bh"].sum()),
                "n_bb_by": int(df["significant_bb_by"].sum()),
                "window_start": window_start,
            },
            basename="majors_full_dual",
        )
    return df


def majors_rolling_screen(
    config: DataConfig | None = None,
    *,
    exchange_id: str | None = None,
    window_days: int = 365,
    step_days: int = 90,
    n_sims: int = 1500,
    seed: int = 42,
    persist: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Rolling EG + dual-null on 6 majors; BY within each window and pooled."""
    config = config or load_config()
    eid = resolve_exchange_id(config, exchange_id)
    store = ParquetStore(config.storage.root, eid, config.ohlcv.timeframe)
    alpha = config.cointegration.pvalue_threshold
    min_obs = min(config.cointegration.min_obs, max(60, window_days - 5))

    frames = {s: load_symbol_frame(store, s) for s in MAJORS}
    t0, t1 = common_time_bounds(frames)
    windows = iter_windows(t0, t1, window_days=window_days, step_days=step_days)

    rows: list[dict] = []
    idx = 0
    for w_start, w_end, s_iso, e_iso in windows:
        sliced = {s: _slice_ohlcv(df, w_start, w_end) for s, df in frames.items()}
        for y_sym, x_sym in MAJOR_PAIRS:
            try:
                row = _run_dual_on_slice(
                    y_sym,
                    x_sym,
                    sliced[y_sym],
                    sliced[x_sym],
                    min_obs=min_obs,
                    price_space=config.cointegration.price_space,
                    n_sims=n_sims,
                    seed=seed + 29 * (idx + 1),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Rolling majors fail %s %s→%s: %s", y_sym, s_iso, e_iso, exc)
                continue
            row.update(
                {
                    "sample": "rolling",
                    "window_start": s_iso,
                    "window_end": e_iso,
                    "window_days": window_days,
                    "significant_raw": row["pvalue_table"] < alpha,
                }
            )
            rows.append(row)
            idx += 1
            logger.info(
                "Majors roll %s %s→%s: table=%.4f rw=%.4f bb=%.4f",
                row["pair_label"],
                s_iso,
                e_iso,
                row["pvalue_table"],
                row["pvalue_mc"],
                row["pvalue_bb"],
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df, {}

    # Per-window BY on p_bb (m=6)
    pieces = []
    for (_, _), g in df.groupby(["window_start", "window_end"]):
        g = g.copy()
        by = apply_benjamini_yekutieli(g, pvalue_col="pvalue_bb", alpha=alpha)
        g["pvalue_bb_by_window"] = by["pvalue_by"].to_numpy()
        g["significant_bb_by_window"] = by["significant_by"].to_numpy()
        bh = apply_benjamini_hochberg(g, pvalue_col="pvalue_bb", alpha=alpha)
        g["pvalue_bb_bh_window"] = bh["pvalue_fdr"].to_numpy()
        g["significant_bb_bh_window"] = bh["significant_fdr"].to_numpy()
        pieces.append(g)
    df = pd.concat(pieces, ignore_index=True)

    # Pooled BY across all pair-windows
    pooled = apply_benjamini_yekutieli(df, pvalue_col="pvalue_bb", alpha=alpha)
    df["pvalue_bb_by_pooled"] = pooled["pvalue_by"]
    df["significant_bb_by_pooled"] = pooled["significant_by"]

    # Hub involvement on raw hits
    hubs = {
        h: hub_involvement_summary(df, hub_symbol=h, significant_col="significant_raw")
        for h in MAJORS
    }
    # Per-pair raw / bb hit rates across windows
    pair_rates = (
        df.groupby("pair_label", as_index=False)
        .agg(
            n_windows=("pair", "count"),
            raw_hit_rate=("significant_raw", "mean"),
            bb_lt_05_rate=("pvalue_bb", lambda s: float((s < alpha).mean())),
            by_window_rate=("significant_bb_by_window", "mean"),
            median_p_bb=("pvalue_bb", "median"),
            median_p_rw=("pvalue_mc", "median"),
        )
        .sort_values("bb_lt_05_rate", ascending=False)
    )

    summary = {
        "n_windows": int(df[["window_start", "window_end"]].drop_duplicates().shape[0]),
        "n_rows": int(len(df)),
        "n_raw_hits": int(df["significant_raw"].sum()),
        "n_bb_lt_05": int((df["pvalue_bb"] < alpha).sum()),
        "n_by_window_sig": int(df["significant_bb_by_window"].sum()),
        "n_by_pooled_sig": int(df["significant_bb_by_pooled"].sum()),
        "hubs": hubs,
    }

    if persist:
        save_scan_results(
            df,
            root=config.storage.root,
            exchange_id=eid,
            timeframe=config.ohlcv.timeframe,
            mode=config.cointegration.persist_mode,
            meta={
                "kind": "majors_rolling_dual",
                "window_days": window_days,
                "step_days": step_days,
                "n_sims": n_sims,
                **{k: v for k, v in summary.items() if k != "hubs"},
            },
            basename=f"majors_rolling_{window_days}d_dual",
        )
        save_scan_results(
            pair_rates,
            root=config.storage.root,
            exchange_id=eid,
            timeframe=config.ohlcv.timeframe,
            mode=config.cointegration.persist_mode,
            meta={"kind": "majors_rolling_pair_rates"},
            basename=f"majors_rolling_{window_days}d_pair_rates",
        )
    return df, {"summary": summary, "pair_rates": pair_rates, "hubs": hubs}


def majors_residual_correlations(
    config: DataConfig | None = None,
    *,
    exchange_id: str | None = None,
    persist: bool = True,
    window_start: str | None = None,
    window_end: str | None = None,
) -> pd.DataFrame:
    """Pairwise residual corr among all 6 major-pair OLS spreads."""
    config = config or load_config()
    eid = resolve_exchange_id(config, exchange_id)
    _panel, corr = candidate_spread_residuals(
        config,
        pairs=MAJOR_PAIRS,
        exchange_id=eid,
        window_start=window_start,
        window_end=window_end,
    )
    corr = corr.copy()
    corr["sample"] = (
        f"{window_start or 'full'}→{window_end or 'end'}"
        if window_start or window_end
        else "full_overlap"
    )
    if persist:
        save_scan_results(
            corr,
            root=config.storage.root,
            exchange_id=eid,
            timeframe=config.ohlcv.timeframe,
            mode=config.cointegration.persist_mode,
            meta={"kind": "majors_residual_corr"},
            basename="majors_residual_corr",
        )
    return corr


def run_majors_full_chain(
    config: DataConfig | None = None,
    *,
    exchange_id: str | None = None,
    n_sims_full: int = 2000,
    n_sims_roll: int = 1500,
    window_days: int = 365,
    step_days: int = 90,
    seed: int = 42,
    persist: bool = True,
) -> dict:
    """Execute full majors validation chain."""
    config = config or load_config()
    full = majors_full_sample_screen(
        config,
        exchange_id=exchange_id,
        n_sims=n_sims_full,
        seed=seed,
        persist=persist,
        window_start="2022-01-01",
    )
    roll, roll_meta = majors_rolling_screen(
        config,
        exchange_id=exchange_id,
        window_days=window_days,
        step_days=step_days,
        n_sims=n_sims_roll,
        seed=seed,
        persist=persist,
    )
    # Residual corr: full overlap + any BY-window survivors' calendar span if present
    corr_full = majors_residual_correlations(
        config, exchange_id=exchange_id, persist=persist
    )
    corr_parts = [corr_full]
    if not roll.empty and roll["significant_bb_by_window"].any():
        hits = roll[roll["significant_bb_by_window"]]
        for _, r in hits[["window_start", "window_end"]].drop_duplicates().iterrows():
            c = majors_residual_correlations(
                config,
                exchange_id=exchange_id,
                persist=False,
                window_start=str(r["window_start"]),
                window_end=str(r["window_end"]),
            )
            corr_parts.append(c)
    corr = pd.concat(corr_parts, ignore_index=True)
    if persist:
        eid = resolve_exchange_id(config, exchange_id)
        save_scan_results(
            corr,
            root=config.storage.root,
            exchange_id=eid,
            timeframe=config.ohlcv.timeframe,
            mode=config.cointegration.persist_mode,
            meta={"kind": "majors_residual_corr_all_samples"},
            basename="majors_residual_corr",
        )
    return {
        "full": full,
        "rolling": roll,
        "rolling_meta": roll_meta,
        "residual_corr": corr,
    }
