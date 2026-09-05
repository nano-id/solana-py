"""BTC + broad-alt-index residual retests for BNB-free rolling FDR candidates.

Also Benjamini–Yekutieli on the 13 EG block-bootstrap p-values (all rows and
unique-pair dedupe).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from pairs_trading.cointegration.engine import evaluate_pair
from pairs_trading.cointegration.fdr import apply_benjamini_yekutieli
from pairs_trading.cointegration.hf_retest import load_rolling_fdr_hits
from pairs_trading.cointegration.neutralize import (
    DEFAULT_ALT_INDEX_EXCLUDE,
    build_btc_alt_residual_frames,
    build_factor_residual_frames,
)
from pairs_trading.cointegration.persist import save_scan_results
from pairs_trading.cointegration.scanner import load_symbol_frame, resolve_exchange_id
from pairs_trading.data.config import DataConfig, load_config
from pairs_trading.data.storage import ParquetStore

logger = logging.getLogger(__name__)

BNB_FREE_CANDIDATE_PAIRS = (
    ("ADA/USDT", "ARB/USDT"),
    ("ADA/USDT", "WLD/USDT"),
    ("ARB/USDT", "WLD/USDT"),
    ("OKB/USDT", "WLD/USDT"),
)


def _iso_to_ms(iso: str) -> int:
    dt = datetime.strptime(str(iso), "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _slice_ohlcv(df: pd.DataFrame, start_ms: int, end_ms: int) -> pd.DataFrame:
    out = df[(df["timestamp"] >= start_ms) & (df["timestamp"] < end_ms)].copy()
    return out.reset_index(drop=True)


def _canon_pair(y: str, x: str) -> str:
    a, b = sorted([y.upper(), x.upper()])
    return f"{a}|{b}"


def apply_by_to_p_bb(
    dual_df: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """BY on all p_bb rows and on unique-pair (min p_bb) representatives."""
    if dual_df.empty or "pvalue_bb" not in dual_df.columns:
        return dual_df, pd.DataFrame(), {}

    all_df = dual_df.copy()
    all_df = apply_benjamini_yekutieli(all_df, pvalue_col="pvalue_bb", alpha=alpha)
    all_df = all_df.rename(
        columns={"pvalue_by": "pvalue_bb_by", "significant_by": "significant_bb_by"}
    )

    # Unique unordered pair: keep the row with smallest p_bb
    tmp = all_df.copy()
    tmp["pair_canon"] = [
        _canon_pair(str(r.y_symbol), str(r.x_symbol)) for r in tmp.itertuples()
    ]
    uniq = (
        tmp.sort_values("pvalue_bb")
        .drop_duplicates("pair_canon", keep="first")
        .reset_index(drop=True)
    )
    # Recompute BY on the unique subset only (m = n_unique)
    uniq_by = apply_benjamini_yekutieli(
        uniq[["pvalue_bb"]].copy(), pvalue_col="pvalue_bb", alpha=alpha
    )
    uniq["pvalue_bb_by_unique"] = uniq_by["pvalue_by"].to_numpy()
    uniq["significant_bb_by_unique"] = uniq_by["significant_by"].to_numpy()

    summary = {
        "n_all": int(len(all_df)),
        "n_all_bb_lt_alpha": int((all_df["pvalue_bb"] < alpha).sum()),
        "n_all_by_sig": int(all_df["significant_bb_by"].sum()),
        "n_unique": int(len(uniq)),
        "n_unique_bb_lt_alpha": int((uniq["pvalue_bb"] < alpha).sum()),
        "n_unique_by_sig": int(uniq["significant_bb_by_unique"].sum()),
        "alpha": alpha,
    }
    return all_df, uniq, summary


def retest_bnb_free_with_residuals(
    config: DataConfig | None = None,
    *,
    exchange_id: str | None = None,
    window_days: tuple[int, ...] = (180, 365),
    persist: bool = True,
) -> pd.DataFrame:
    """For BNB-free FDR windows: raw / BTC-resid / BTC+alt-resid EG p-values."""
    config = config or load_config()
    eid = resolve_exchange_id(config, exchange_id)
    store = ParquetStore(config.storage.root, eid, config.ohlcv.timeframe)
    hits = load_rolling_fdr_hits(
        config,
        exchange_id=eid,
        window_days=window_days,
        timeframe=config.ohlcv.timeframe,
    )
    if hits.empty:
        return pd.DataFrame()

    cand_set = {tuple(sorted(p)) for p in BNB_FREE_CANDIDATE_PAIRS}
    mask = hits.apply(
        lambda r: tuple(sorted([str(r["y_symbol"]), str(r["x_symbol"])])) in cand_set,
        axis=1,
    )
    cand = hits.loc[mask].copy()
    if cand.empty:
        return pd.DataFrame()

    need_syms = sorted(
        {
            *cand["y_symbol"].astype(str),
            *cand["x_symbol"].astype(str),
            "BTC/USDT",
            *DEFAULT_ALT_INDEX_EXCLUDE,
        }
    )
    # Full universe for alt index
    for stem in store.list_symbols():
        parts = stem.split("_", 1)
        if len(parts) == 2:
            need_syms.append(f"{parts[0]}/{parts[1]}")
    need_syms = sorted(set(need_syms))

    raw_frames: dict[str, pd.DataFrame] = {}
    for sym in need_syms:
        try:
            raw_frames[sym] = load_symbol_frame(store, sym)
        except FileNotFoundError:
            continue

    rows: list[dict] = []
    for _, hit in cand.iterrows():
        y_sym, x_sym = str(hit["y_symbol"]), str(hit["x_symbol"])
        w0, w1 = _iso_to_ms(hit["window_start"]), _iso_to_ms(hit["window_end"])
        window_days_i = int(hit.get("window_days", 180))
        min_obs = max(60, window_days_i - 5)

        # Slice full universe to this window for fair residualization
        sliced = {
            s: _slice_ohlcv(df, w0, w1)
            for s, df in raw_frames.items()
            if len(_slice_ohlcv(df, w0, w1)) >= min_obs
        }
        if y_sym not in sliced or x_sym not in sliced or "BTC/USDT" not in sliced:
            logger.warning("Skip %s|%s — missing window data", y_sym, x_sym)
            continue

        modes: list[tuple[str, dict[str, pd.DataFrame], str]] = []

        # raw log EG
        modes.append(("raw", sliced, "log"))

        # BTC residual only
        try:
            btc_res, _ = build_factor_residual_frames(
                sliced, factor_symbol="BTC/USDT", min_obs=min_obs
            )
            modes.append(("btc_residual", btc_res, "level"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("BTC residual failed %s|%s: %s", y_sym, x_sym, exc)

        # BTC + alt index
        try:
            btc_alt, meta, _idx, members = build_btc_alt_residual_frames(
                sliced,
                btc_symbol="BTC/USDT",
                exclude_from_alt_index=DEFAULT_ALT_INDEX_EXCLUDE,
                min_obs=min_obs,
                symbols=[y_sym, x_sym],
            )
            modes.append(("btc_alt_residual", btc_alt, "level"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("BTC+alt residual failed %s|%s: %s", y_sym, x_sym, exc)
            meta, members = {}, []

        base = {
            "window_start": hit["window_start"],
            "window_end": hit["window_end"],
            "window_days": window_days_i,
            "y_symbol": y_sym,
            "x_symbol": x_sym,
            "pair": f"{y_sym}|{x_sym}",
            "daily_pvalue": float(hit["pvalue"]),
            "daily_pvalue_fdr": float(hit.get("pvalue_fdr", float("nan"))),
            "alt_index_n_members": len(members) if members else np.nan,
        }

        for mode, frames, price_space in modes:
            if y_sym not in frames or x_sym not in frames:
                continue
            try:
                eg = evaluate_pair(
                    y_sym,
                    x_sym,
                    frames[y_sym],
                    frames[x_sym],
                    min_obs=min_obs,
                    price_space=price_space,  # type: ignore[arg-type]
                )
            except ValueError as exc:
                logger.warning("%s EG failed %s|%s: %s", mode, y_sym, x_sym, exc)
                continue
            row = {
                **base,
                "preprocess": mode,
                "pvalue": eg.pvalue,
                "test_statistic": eg.test_statistic,
                "hedge_ratio": eg.hedge_ratio,
                "half_life": eg.half_life,
                "n_obs": eg.n_obs,
            }
            if mode == "btc_alt_residual" and y_sym in meta and x_sym in meta:
                row["y_beta_btc"] = meta[y_sym].get("beta_btc", float("nan"))
                row["y_beta_alt"] = meta[y_sym].get("beta_alt_index", float("nan"))
                row["x_beta_btc"] = meta[x_sym].get("beta_btc", float("nan"))
                row["x_beta_alt"] = meta[x_sym].get("beta_alt_index", float("nan"))
            rows.append(row)
            logger.info(
                "%s %s|%s %s→%s: p=%.4g HL=%.2f",
                mode,
                y_sym,
                x_sym,
                hit["window_start"],
                hit["window_end"],
                eg.pvalue,
                eg.half_life,
            )

    out = pd.DataFrame(rows)
    if persist and not out.empty:
        save_scan_results(
            out,
            root=config.storage.root,
            exchange_id=eid,
            timeframe=config.ohlcv.timeframe,
            mode=config.cointegration.persist_mode,
            meta={
                "kind": "bnb_free_residual_retest",
                "candidates": [f"{a}|{b}" for a, b in BNB_FREE_CANDIDATE_PAIRS],
                "n_rows": len(out),
            },
            basename="bnb_free_residual_retest",
        )
    return out


def run_p_bb_by_and_residual_report(
    config: DataConfig | None = None,
    *,
    exchange_id: str | None = None,
    persist: bool = True,
) -> dict[str, pd.DataFrame | dict]:
    """Load EG dual-null artifact → BY on p_bb; run residual retests."""
    config = config or load_config()
    eid = resolve_exchange_id(config, exchange_id)
    path = (
        config.storage.root
        / eid
        / "cointegration"
        / f"eg_dual_null_rolling_fdr_latest_{config.ohlcv.timeframe}.parquet"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing EG dual-null artifact: {path}")
    dual = pd.read_parquet(path)
    all_by, uniq_by, by_summary = apply_by_to_p_bb(
        dual, alpha=config.cointegration.pvalue_threshold
    )
    resid = retest_bnb_free_with_residuals(
        config, exchange_id=eid, persist=persist
    )

    if persist:
        if not all_by.empty:
            save_scan_results(
                all_by,
                root=config.storage.root,
                exchange_id=eid,
                timeframe=config.ohlcv.timeframe,
                mode=config.cointegration.persist_mode,
                meta={"kind": "eg_p_bb_by_all", **by_summary},
                basename="eg_p_bb_by_all",
            )
        if not uniq_by.empty:
            save_scan_results(
                uniq_by,
                root=config.storage.root,
                exchange_id=eid,
                timeframe=config.ohlcv.timeframe,
                mode=config.cointegration.persist_mode,
                meta={"kind": "eg_p_bb_by_unique", **by_summary},
                basename="eg_p_bb_by_unique",
            )

    return {
        "p_bb_by_all": all_by,
        "p_bb_by_unique": uniq_by,
        "by_summary": by_summary,
        "residual_retest": resid,
    }
