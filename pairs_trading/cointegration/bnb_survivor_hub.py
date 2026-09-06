"""Fixed-window hub/placebo dual-null + BY for BNB FDR survivors."""

from __future__ import annotations

import logging
from dataclasses import asdict

import pandas as pd

from pairs_trading.cointegration.eg_dual_null import dual_null_eg_pair
from pairs_trading.cointegration.engine import align_close_prices, evaluate_pair
from pairs_trading.cointegration.fdr import apply_benjamini_yekutieli
from pairs_trading.cointegration.persist import save_scan_results
from pairs_trading.cointegration.scanner import load_symbol_frame, resolve_exchange_id
from pairs_trading.cointegration.wld_hub import (
    WLD_PLACEBOS,
    _iso_to_ms,
    _slice_ohlcv,
    candidate_spread_residuals,
)
from pairs_trading.data.config import DataConfig, load_config
from pairs_trading.data.storage import ParquetStore

logger = logging.getLogger(__name__)

# Same economically distant set used for WLD / BNB placebo checks.
BNB_PLACEBOS = WLD_PLACEBOS

# Windows where BNB|BTC / BNB|UNI survived EG dual-null BY.
BNB_SURVIVOR_WINDOWS = (
    ("2024-03-08", "2025-03-08", "bnb_btc_uni_365"),  # BNB|BTC + BNB|UNI
    ("2024-06-06", "2025-06-06", "bnb_btc_365"),  # BNB|BTC
)

BNB_SURVIVOR_PAIRS = (
    ("BNB/USDT", "BTC/USDT"),
    ("BNB/USDT", "UNI/USDT"),
)


def hub_placebo_dual_null_fixed_windows(
    config: DataConfig | None = None,
    *,
    hub_symbol: str = "BNB/USDT",
    placebos: tuple[str, ...] | list[str] | None = None,
    windows: tuple[tuple[str, str, str], ...] = BNB_SURVIVOR_WINDOWS,
    also_pairs: tuple[tuple[str, str], ...] = BNB_SURVIVOR_PAIRS,
    exchange_id: str | None = None,
    n_sims: int = 1500,
    seed: int = 42,
    persist: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """EG + dual-null for hub×placebo (and optional survivor pairs) in fixed windows.

    Applies BY to p_bb within each window's placebo family (m = n_placebos),
    and also a pooled BY across all hub×placebo rows.
    """
    config = config or load_config()
    eid = resolve_exchange_id(config, exchange_id)
    store = ParquetStore(config.storage.root, eid, config.ohlcv.timeframe)
    placebos = list(placebos or BNB_PLACEBOS)
    alpha = config.cointegration.pvalue_threshold

    need = {hub_symbol, *placebos}
    for y, x in also_pairs:
        need.add(y)
        need.add(x)
    frames: dict[str, pd.DataFrame] = {}
    for sym in sorted(need):
        try:
            frames[sym] = load_symbol_frame(store, sym)
        except FileNotFoundError:
            logger.warning("Missing %s", sym)

    if hub_symbol not in frames:
        raise FileNotFoundError(f"Hub {hub_symbol} missing")

    rows: list[dict] = []
    idx = 0
    for w_start, w_end, w_label in windows:
        w0, w1 = _iso_to_ms(w_start), _iso_to_ms(w_end)
        window_days = int(round((w1 - w0) / 86_400_000))
        min_obs = max(60, window_days - 5)

        # Hub × placebos
        targets: list[tuple[str, str, str]] = [
            (hub_symbol, p, "placebo") for p in placebos if p in frames
        ]
        # Survivor pairs for side-by-side comparison
        for y, x in also_pairs:
            if y in frames and x in frames:
                targets.append((y, x, "survivor"))

        for y_sym, x_sym, kind in targets:
            y_df = _slice_ohlcv(frames[y_sym], w0, w1)
            x_df = _slice_ohlcv(frames[x_sym], w0, w1)
            try:
                hit = evaluate_pair(
                    y_sym,
                    x_sym,
                    y_df,
                    x_df,
                    min_obs=min_obs,
                    price_space=config.cointegration.price_space,
                )
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
                    seed=seed + 19 * (idx + 1),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Fail %s|%s %s: %s", y_sym, x_sym, w_label, exc)
                continue
            idx += 1
            row = {
                **asdict(hit),
                "pair": f"{y_sym}|{x_sym}",
                "kind": kind,
                "window_start": w_start,
                "window_end": w_end,
                "window_label": w_label,
                "window_days": window_days,
                "pvalue_table": hit.pvalue,
                "significant_raw": hit.pvalue < alpha,
                **dual,
            }
            rows.append(row)
            logger.info(
                "BNB window %s %s (%s): table=%.4f rw=%.4f bb=%.4f",
                w_label,
                row["pair"],
                kind,
                hit.pvalue,
                dual["pvalue_mc"],
                dual["pvalue_bb"],
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df, pd.DataFrame(), {}

    # BY per window on placebo rows only
    by_pieces: list[pd.DataFrame] = []
    for w_label, g in df.groupby("window_label"):
        placebo = g[g["kind"] == "placebo"].copy()
        surv = g[g["kind"] == "survivor"].copy()
        if not placebo.empty:
            placebo = apply_benjamini_yekutieli(
                placebo, pvalue_col="pvalue_bb", alpha=alpha
            )
            placebo = placebo.rename(
                columns={
                    "pvalue_by": "pvalue_bb_by_window",
                    "significant_by": "significant_bb_by_window",
                }
            )
        if not surv.empty:
            surv["pvalue_bb_by_window"] = float("nan")
            surv["significant_bb_by_window"] = False
        by_pieces.append(pd.concat([placebo, surv], ignore_index=True))
    per_window = pd.concat(by_pieces, ignore_index=True)

    # Pooled BY on all placebo p_bb
    placebo_all = per_window[per_window["kind"] == "placebo"].copy()
    pooled = apply_benjamini_yekutieli(
        placebo_all, pvalue_col="pvalue_bb", alpha=alpha
    )
    placebo_all = placebo_all.copy()
    placebo_all["pvalue_bb_by_pooled"] = pooled["pvalue_by"].to_numpy()
    placebo_all["significant_bb_by_pooled"] = pooled["significant_by"].to_numpy()

    # Merge pooled cols back
    per_window = per_window.merge(
        placebo_all[
            ["pair", "window_label", "pvalue_bb_by_pooled", "significant_bb_by_pooled"]
        ],
        on=["pair", "window_label"],
        how="left",
    )

    summary = {
        "hub_symbol": hub_symbol,
        "n_placebo_rows": int((per_window["kind"] == "placebo").sum()),
        "n_placebo_raw_hits": int(
            per_window.loc[per_window["kind"] == "placebo", "significant_raw"].sum()
        ),
        "n_placebo_bb_lt_05": int(
            (
                (per_window["kind"] == "placebo")
                & (per_window["pvalue_bb"] < alpha)
            ).sum()
        ),
        "n_placebo_by_window_sig": int(
            per_window["significant_bb_by_window"].fillna(False).sum()
        ),
        "n_placebo_by_pooled_sig": int(
            per_window["significant_bb_by_pooled"].fillna(False).sum()
        ),
        "placebo_raw_hit_rate": float(
            per_window.loc[per_window["kind"] == "placebo", "significant_raw"].mean()
        ),
        "placebo_bb_lt_05_rate": float(
            (
                per_window.loc[per_window["kind"] == "placebo", "pvalue_bb"] < alpha
            ).mean()
        ),
    }
    # Per-window placebo hit rates
    for w_label, g in per_window[per_window["kind"] == "placebo"].groupby("window_label"):
        summary[f"{w_label}_raw_hit_rate"] = float(g["significant_raw"].mean())
        summary[f"{w_label}_bb_lt_05_rate"] = float((g["pvalue_bb"] < alpha).mean())
        summary[f"{w_label}_by_window_sig"] = int(g["significant_bb_by_window"].sum())

    if persist and not per_window.empty:
        save_scan_results(
            per_window,
            root=config.storage.root,
            exchange_id=eid,
            timeframe=config.ohlcv.timeframe,
            mode=config.cointegration.persist_mode,
            meta={"kind": "bnb_survivor_placebo_dual", "n_sims": n_sims, **summary},
            basename="bnb_survivor_placebo_dual",
        )

    return per_window, placebo_all, summary


def correlate_bnb_survivor_residuals(
    config: DataConfig | None = None,
    *,
    exchange_id: str | None = None,
    persist: bool = True,
) -> pd.DataFrame:
    """Correlate BNB-BTC and BNB-UNI OLS spreads on survivor windows + full."""
    config = config or load_config()
    eid = resolve_exchange_id(config, exchange_id)
    samples = [
        ("full", None, None),
        ("2024-03→2025-03", "2024-03-08", "2025-03-08"),
        ("2024-06→2025-06", "2024-06-06", "2025-06-06"),
    ]
    frames: list[pd.DataFrame] = []
    for label, ws, we in samples:
        _panel, corr = candidate_spread_residuals(
            config,
            pairs=BNB_SURVIVOR_PAIRS,
            exchange_id=eid,
            window_start=ws,
            window_end=we,
        )
        corr = corr.copy()
        corr["sample"] = label
        frames.append(corr)
        off = corr[corr["pair_a"] != corr["pair_b"]]
        logger.info(
            "BNB survivor residual corr [%s]: level=%.3f diff=%.3f",
            label,
            off["corr_level"].abs().mean(),
            off["corr_diff"].abs().mean(),
        )
    out = pd.concat(frames, ignore_index=True)
    if persist:
        save_scan_results(
            out,
            root=config.storage.root,
            exchange_id=eid,
            timeframe=config.ohlcv.timeframe,
            mode=config.cointegration.persist_mode,
            meta={"kind": "bnb_survivor_residual_corr"},
            basename="bnb_survivor_residual_corr",
        )
    return out
