"""IJSRA 2026 claimed setup × Rule 5d (dual-null + BY) stress test.

Paper (Park, IJSRA 2026, 18(02):516-529):
  BTC/ETH/LTC/XRP daily, 2022-01-01 → 2024-10-31
  Engle–Granger + Johansen only (no multi-window / dual-null / BY)
  BTC–ETH claim: EG p≈0.0003; ann. 16.34% gross / 14.89% @ 0.10% TC
  Signals: z60, entry |z|≥2, exit zero-cross, stop |z|≥3

Reproduce on OKX daily, then apply Rule 5d:
  dual-null (independent RW + block-bootstrap) + Benjamini–Yekutieli
  on (a) EG across 6 pairs and (b) strategy ann. net on 4 traded pairs.

interpretation_allowed=false — control only.
"""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from pairs_trading.cointegration.eg_dual_null import dual_null_eg_pair
from pairs_trading.cointegration.fdr import apply_benjamini_yekutieli
from pairs_trading.cointegration.johansen_mc import (
    block_bootstrap_levels,
    default_block_size,
)
from pairs_trading.cointegration.persist import save_scan_results
from pairs_trading.cointegration.scanner import load_symbol_frame, resolve_exchange_id
from pairs_trading.data.config import DataConfig, load_config
from pairs_trading.data.storage import ParquetStore

logger = logging.getLogger(__name__)

PAPER_START = "2022-01-01"
PAPER_END_EXCLUSIVE = "2024-11-01"  # through 2024-10-31
PAPER_ASSETS = ("BTC/USDT", "ETH/USDT", "LTC/USDT", "XRP/USDT")
PAPER_PAIRS = tuple(itertools.combinations(PAPER_ASSETS, 2))

Z_WINDOW = 60
ENTRY_Z = 2.0
STOP_Z = 3.0
TC_PER_TRADE = 0.001  # 0.10% per trade (paper Table 8)


def _iso_to_ms(iso: str) -> int:
    dt = datetime.strptime(str(iso), "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _slice_ohlcv(df: pd.DataFrame, start_ms: int, end_ms: int) -> pd.DataFrame:
    out = df[(df["timestamp"] >= start_ms) & (df["timestamp"] < end_ms)].copy()
    return out.reset_index(drop=True)


def _pair_label(y: str, x: str) -> str:
    a, b = sorted([y.split("/")[0], x.split("/")[0]])
    return f"{a}-{b}"


def _orient(a: str, b: str) -> tuple[str, str]:
    """y = non-hub, x = BTC (else ETH) hub — matches paper BTC-ETH β narrative."""
    pair = {a, b}
    if "BTC/USDT" in pair:
        x = "BTC/USDT"
        y = next(s for s in pair if s != x)
        return y, x
    if "ETH/USDT" in pair:
        x = "ETH/USDT"
        y = next(s for s in pair if s != x)
        return y, x
    return a, b


def load_paper_window_frames(
    config: DataConfig | None = None,
    *,
    exchange_id: str | None = None,
) -> tuple[str, dict[str, pd.DataFrame]]:
    config = config or load_config()
    eid = resolve_exchange_id(config, exchange_id)
    store = ParquetStore(config.storage.root, eid, config.ohlcv.timeframe)
    w0, w1 = _iso_to_ms(PAPER_START), _iso_to_ms(PAPER_END_EXCLUSIVE)
    frames = {
        s: _slice_ohlcv(load_symbol_frame(store, s), w0, w1) for s in PAPER_ASSETS
    }
    return eid, frames


def aligned_closes(
    frames: dict[str, pd.DataFrame], y_sym: str, x_sym: str
) -> pd.DataFrame:
    y = frames[y_sym][["timestamp", "close"]].rename(columns={"close": "y"})
    x = frames[x_sym][["timestamp", "close"]].rename(columns={"close": "x"})
    m = y.merge(x, on="timestamp", how="inner").dropna().sort_values("timestamp")
    return m.reset_index(drop=True)


def eg_paper_pair(
    frames: dict[str, pd.DataFrame],
    y_sym: str,
    x_sym: str,
    *,
    price_space: Literal["log", "level"] = "log",
) -> dict[str, Any]:
    m = aligned_closes(frames, y_sym, x_sym)
    y = m["y"].to_numpy(dtype=float)
    x = m["x"].to_numpy(dtype=float)
    if price_space == "log":
        y = np.log(y)
        x = np.log(x)
    ols = sm.OLS(y, sm.add_constant(x)).fit()
    alpha = float(ols.params[0])
    beta = float(ols.params[1])
    resid = y - (alpha + beta * x)
    score, pvalue, crit = coint(y, x, autolag="aic")
    return {
        "pair_label": _pair_label(y_sym, x_sym),
        "y_symbol": y_sym,
        "x_symbol": x_sym,
        "price_space": price_space,
        "n_obs": int(len(m)),
        "beta": beta,
        "alpha": alpha,
        "test_statistic": float(score),
        "pvalue_table": float(pvalue),
        "crit_5pct": float(crit[1]) if crit is not None and len(crit) > 1 else None,
        "resid_std": float(np.std(resid, ddof=1)),
    }


def johansen_paper_system(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    merged: pd.DataFrame | None = None
    for sym in PAPER_ASSETS:
        piece = frames[sym][["timestamp", "close"]].rename(columns={"close": sym})
        piece[sym] = np.log(piece[sym].astype(float))
        merged = piece if merged is None else merged.merge(piece, on="timestamp", how="inner")
    assert merged is not None
    merged = merged.dropna().sort_values("timestamp")
    data = merged[list(PAPER_ASSETS)].to_numpy()
    res = coint_johansen(data, det_order=0, k_ar_diff=1)
    trace = np.asarray(res.lr1, dtype=float)
    cvt = np.asarray(res.cvt, dtype=float)
    maxeig = np.asarray(res.lr2, dtype=float)
    cve = np.asarray(res.cvm, dtype=float)
    return {
        "n_obs": int(len(merged)),
        "trace_r0": float(trace[0]),
        "trace_cv95_r0": float(cvt[0, 1]),
        "trace_rejects_r0_5pct": bool(trace[0] > cvt[0, 1]),
        "maxeig_r0": float(maxeig[0]),
        "maxeig_cv95_r0": float(cve[0, 1]),
        "maxeig_rejects_r0_5pct": bool(maxeig[0] > cve[0, 1]),
        "n_coint_trace_5pct": int(np.sum(trace > cvt[:, 1])),
        "paper_claimed_trace_r0": 78.456,
        "paper_claimed_maxeig_r0": 42.345,
    }


def _annualize(total_return: float, n_days: int) -> float:
    if n_days <= 0 or not np.isfinite(total_return):
        return float("nan")
    return float((1.0 + total_return) ** (365.0 / n_days) - 1.0)


@dataclass(frozen=True)
class PaperTradeResult:
    pair_label: str
    y_symbol: str
    x_symbol: str
    n_obs: int
    n_trades: int
    win_rate: float
    total_gross: float
    total_cost: float
    total_net: float
    ann_gross: float
    ann_net: float
    ann_vol: float
    sharpe_rf0: float
    max_dd: float
    avg_hold_days: float
    beta: float
    price_space: str


def paper_pairs_backtest(
    frames: dict[str, pd.DataFrame],
    y_sym: str,
    x_sym: str,
    *,
    price_space: Literal["log", "level"] = "log",
    z_window: int = Z_WINDOW,
    entry_z: float = ENTRY_Z,
    stop_z: float = STOP_Z,
    tc_per_trade: float = TC_PER_TRADE,
    beta: float | None = None,
    alpha: float | None = None,
) -> PaperTradeResult:
    """Full-sample β (paper style), z60, entry±2, exit zero-cross, stop±3."""
    m = aligned_closes(frames, y_sym, x_sym)
    y = m["y"].to_numpy(dtype=float)
    x = m["x"].to_numpy(dtype=float)
    if price_space == "log":
        y = np.log(y)
        x = np.log(x)
    if beta is None or alpha is None:
        ols = sm.OLS(y, sm.add_constant(x)).fit()
        alpha = float(ols.params[0])
        beta = float(ols.params[1])
    assert alpha is not None and beta is not None
    resid = y - (alpha + beta * x)

    s = pd.Series(resid)
    mu = s.rolling(z_window, min_periods=z_window).mean()
    sd = s.rolling(z_window, min_periods=z_window).std(ddof=0)
    z = ((s - mu) / sd.replace(0.0, np.nan)).to_numpy()

    pos = 0
    daily = np.zeros(len(resid))
    trades_pnl: list[float] = []
    holds: list[int] = []
    entry_i: int | None = None
    trade_gross = 0.0
    total_cost = 0.0

    for i in range(1, len(resid)):
        zi = z[i]
        if pos != 0:
            daily[i] = pos * (resid[i] - resid[i - 1])
            trade_gross += daily[i]
        if not np.isfinite(zi):
            continue
        if pos == 0:
            if zi <= -entry_z:
                pos = 1
                entry_i = i
                trade_gross = 0.0
                total_cost += tc_per_trade
            elif zi >= entry_z:
                pos = -1
                entry_i = i
                trade_gross = 0.0
                total_cost += tc_per_trade
        else:
            cross_zero = (pos == 1 and zi >= 0.0) or (pos == -1 and zi <= 0.0)
            if cross_zero or abs(zi) >= stop_z:
                total_cost += tc_per_trade
                trades_pnl.append(trade_gross - 2.0 * tc_per_trade)
                if entry_i is not None:
                    holds.append(i - entry_i)
                pos = 0
                entry_i = None
                trade_gross = 0.0

    if pos != 0 and entry_i is not None:
        total_cost += tc_per_trade
        trades_pnl.append(trade_gross - 2.0 * tc_per_trade)
        holds.append(len(resid) - 1 - entry_i)

    equity = np.cumsum(daily)
    total_gross = float(equity[-1]) if len(equity) else 0.0
    total_net = total_gross - total_cost
    n_days = int(len(resid))
    ann_g = _annualize(total_gross, n_days)
    ann_n = _annualize(total_net, n_days)
    vol = float(np.std(daily, ddof=1) * np.sqrt(365)) if len(daily) > 2 else float("nan")
    sharpe = float(ann_n / vol) if vol and np.isfinite(vol) and vol > 0 else float("nan")
    peak = np.maximum.accumulate(equity) if len(equity) else equity
    dd = equity - peak if len(equity) else equity
    max_dd = float(dd.min()) if len(dd) else 0.0
    wins = sum(1 for t in trades_pnl if t > 0)
    n_tr = len(trades_pnl)
    return PaperTradeResult(
        pair_label=_pair_label(y_sym, x_sym),
        y_symbol=y_sym,
        x_symbol=x_sym,
        n_obs=n_days,
        n_trades=n_tr,
        win_rate=float(wins / n_tr) if n_tr else float("nan"),
        total_gross=total_gross,
        total_cost=total_cost,
        total_net=total_net,
        ann_gross=ann_g,
        ann_net=ann_n,
        ann_vol=vol,
        sharpe_rf0=sharpe,
        max_dd=max_dd,
        avg_hold_days=float(np.mean(holds)) if holds else float("nan"),
        beta=float(beta),
        price_space=price_space,
    )


def _right_tail_pvalue(observed: float, nulls: np.ndarray) -> float:
    valid = nulls[np.isfinite(nulls)]
    if len(valid) == 0 or not np.isfinite(observed):
        return 1.0
    n_ge = int(np.sum(valid >= observed))
    return float((1.0 + n_ge) / (len(valid) + 1.0))


def _ann_net_from_log_paths(
    log_y: np.ndarray,
    log_x: np.ndarray,
    *,
    tc_per_trade: float = TC_PER_TRADE,
) -> float:
    ols = sm.OLS(log_y, sm.add_constant(log_x)).fit()
    alpha = float(ols.params[0])
    beta = float(ols.params[1])
    resid = log_y - (alpha + beta * log_x)
    s = pd.Series(resid)
    mu = s.rolling(Z_WINDOW, min_periods=Z_WINDOW).mean()
    sd = s.rolling(Z_WINDOW, min_periods=Z_WINDOW).std(ddof=0)
    z = ((s - mu) / sd.replace(0.0, np.nan)).to_numpy()
    pos = 0
    daily = np.zeros(len(resid))
    total_cost = 0.0
    for i in range(1, len(resid)):
        zi = z[i]
        if pos != 0:
            daily[i] = pos * (resid[i] - resid[i - 1])
        if not np.isfinite(zi):
            continue
        if pos == 0:
            if zi <= -ENTRY_Z or zi >= ENTRY_Z:
                pos = 1 if zi <= -ENTRY_Z else -1
                total_cost += tc_per_trade
        else:
            cross_zero = (pos == 1 and zi >= 0.0) or (pos == -1 and zi <= 0.0)
            if cross_zero or abs(zi) >= STOP_Z:
                total_cost += tc_per_trade
                pos = 0
    if pos != 0:
        total_cost += tc_per_trade
    return _annualize(float(np.sum(daily) - total_cost), len(resid))


def dual_null_strategy_ann_net(
    log_y: np.ndarray,
    log_x: np.ndarray,
    *,
    observed_ann_net: float,
    n_sims: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(log_y)
    log_yx = np.column_stack([log_y, log_x])
    dy = np.diff(log_y)
    dx = np.diff(log_x)
    sy = float(np.std(dy)) if len(dy) else 0.01
    sx = float(np.std(dx)) if len(dx) else 0.01

    rw_stats = np.empty(n_sims, dtype=float)
    for i in range(n_sims):
        shocks = rng.normal(size=(n, 2))
        shocks[:, 0] *= sy
        shocks[:, 1] *= sx
        paths = np.cumsum(shocks, axis=0)
        paths[:, 0] += log_y[0]
        paths[:, 1] += log_x[0]
        try:
            rw_stats[i] = _ann_net_from_log_paths(paths[:, 0], paths[:, 1])
        except Exception:  # noqa: BLE001
            rw_stats[i] = np.nan

    bb_stats = np.empty(n_sims, dtype=float)
    bs = default_block_size(n - 1)
    for i in range(n_sims):
        try:
            paths = block_bootstrap_levels(log_yx, rng, block_size=bs)
            bb_stats[i] = _ann_net_from_log_paths(paths[:, 0], paths[:, 1])
        except Exception:  # noqa: BLE001
            bb_stats[i] = np.nan

    p_rw = _right_tail_pvalue(observed_ann_net, rw_stats)
    p_bb = _right_tail_pvalue(observed_ann_net, bb_stats)
    return {
        "observed_ann_net": float(observed_ann_net),
        "pvalue_rw": p_rw,
        "pvalue_bb": p_bb,
        "pvalue_dual": float(max(p_rw, p_bb)),
        "rw_null_mean": float(np.nanmean(rw_stats)),
        "rw_null_p95": float(np.nanquantile(rw_stats, 0.95)),
        "bb_null_mean": float(np.nanmean(bb_stats)),
        "bb_null_p95": float(np.nanquantile(bb_stats, 0.95)),
        "block_size": int(bs),
        "n_sims": int(n_sims),
        "n_valid_rw": int(np.isfinite(rw_stats).sum()),
        "n_valid_bb": int(np.isfinite(bb_stats).sum()),
    }


def run_ijsra_rule5d_stress(
    config: DataConfig | None = None,
    *,
    exchange_id: str | None = None,
    n_sims_eg: int = 500,
    n_sims_pnl: int = 200,
    seed: int = 42,
    persist: bool = True,
) -> dict[str, Any]:
    """Reproduce IJSRA claimed setup on OKX, then apply Rule 5d dual-null + BY."""
    eid, frames = load_paper_window_frames(config, exchange_id=exchange_id)
    n_bars = {s: len(df) for s, df in frames.items()}

    eg_log_rows: list[dict] = []
    eg_level_rows: list[dict] = []
    for i, (a, b) in enumerate(PAPER_PAIRS):
        y_s, x_s = _orient(a, b)
        row_log = eg_paper_pair(frames, y_s, x_s, price_space="log")
        row_lvl = eg_paper_pair(frames, y_s, x_s, price_space="level")
        m = aligned_closes(frames, y_s, x_s)
        dual = dual_null_eg_pair(
            np.log(m["y"].to_numpy(dtype=float)),
            np.log(m["x"].to_numpy(dtype=float)),
            n_sims=n_sims_eg,
            seed=seed + 17 * (i + 1),
        )
        row_log.update(
            {
                "pvalue_rw": dual["pvalue_mc"],
                "pvalue_bb": dual["pvalue_bb"],
                "pvalue_dual": float(max(dual["pvalue_mc"], dual["pvalue_bb"])),
                "eg_dual_stat": dual["test_statistic"],
                "n_sims_eg": n_sims_eg,
            }
        )
        eg_log_rows.append(row_log)
        eg_level_rows.append(row_lvl)

    eg_df = pd.DataFrame(eg_log_rows)
    eg_df = apply_benjamini_yekutieli(eg_df, pvalue_col="pvalue_dual", alpha=0.05)
    eg_df["significant_table_5pct"] = eg_df["pvalue_table"] < 0.05
    eg_df["survives_rule5d_coint"] = eg_df["significant_by"].astype(bool)

    joh = johansen_paper_system(frames)

    traded_specs = [
        ("ETH/USDT", "BTC/USDT"),
        ("LTC/USDT", "ETH/USDT"),
        ("LTC/USDT", "BTC/USDT"),
        ("XRP/USDT", "ETH/USDT"),
    ]
    trade_rows: list[dict] = []
    for j, (y_s, x_s) in enumerate(traded_specs):
        bt = paper_pairs_backtest(frames, y_s, x_s, price_space="log")
        m = aligned_closes(frames, y_s, x_s)
        null = dual_null_strategy_ann_net(
            np.log(m["y"].to_numpy(dtype=float)),
            np.log(m["x"].to_numpy(dtype=float)),
            observed_ann_net=bt.ann_net,
            n_sims=n_sims_pnl,
            seed=seed + 101 * (j + 1),
        )
        trade_rows.append(
            {
                **asdict(bt),
                **null,
                "paper_claimed_ann_net": 0.1489 if bt.pair_label == "BTC-ETH" else None,
                "paper_claimed_ann_gross": 0.1634 if bt.pair_label == "BTC-ETH" else None,
            }
        )

    trade_df = pd.DataFrame(trade_rows)
    trade_df = apply_benjamini_yekutieli(trade_df, pvalue_col="pvalue_dual", alpha=0.05)
    trade_df["survives_rule5d_pnl"] = trade_df["significant_by"].astype(bool)

    btc_eth_eg = eg_df.loc[eg_df["pair_label"] == "BTC-ETH"].iloc[0].to_dict()
    btc_eth_tr = trade_df.loc[trade_df["pair_label"] == "BTC-ETH"].iloc[0].to_dict()
    btc_eth_level = next(r for r in eg_level_rows if r["pair_label"] == "BTC-ETH")

    n_coint_survive = int(eg_df["survives_rule5d_coint"].sum())
    n_pnl_survive = int(trade_df["survives_rule5d_pnl"].sum())
    verdict = (
        "PAPER_CLAIM_FAILS_RULE5D"
        if n_coint_survive == 0 and n_pnl_survive == 0
        else "PARTIAL_SURVIVAL"
    )

    report: dict[str, Any] = {
        "step": "ijsra_2026_rule5d_stress",
        "status": "complete",
        "interpretation_allowed": False,
        "exchange_id": eid,
        "paper": {
            "citation": (
                "Park, T. (2026). Statistical Arbitrage Strategies Using Cointegration "
                "Analysis in Cryptocurrency Markets. IJSRA, 18(02), 516-529. "
                "https://doi.org/10.30574/ijsra.2026.18.2.0283"
            ),
            "window": "2022-01-01 → 2024-10-31",
            "assets": list(PAPER_ASSETS),
            "claimed_btc_eth_eg_p": 0.0003,
            "claimed_btc_eth_beta": 0.0587,
            "claimed_ann_gross": 0.1634,
            "claimed_ann_net_10bps": 0.1489,
            "claimed_sharpe": 2.45,
            "methodology_notes": [
                "standard EG + Johansen only — no multi-window, no dual-null, no BY",
                "z60 / entry±2 / exit zero-cross / stop±3 / TC 0.10% per trade",
            ],
        },
        "data_coverage": n_bars,
        "eg_log": eg_df.to_dict(orient="records"),
        "eg_level_btc_eth": btc_eth_level,
        "johansen_4asset": joh,
        "trading": trade_df.to_dict(orient="records"),
        "btc_eth_focus": {
            "eg_pvalue_table": btc_eth_eg["pvalue_table"],
            "eg_pvalue_dual": btc_eth_eg["pvalue_dual"],
            "eg_pvalue_by": btc_eth_eg["pvalue_by"],
            "eg_survives_rule5d": bool(btc_eth_eg["survives_rule5d_coint"]),
            "eg_table_rejects_5pct": bool(btc_eth_eg["pvalue_table"] < 0.05),
            "level_beta_okx": btc_eth_level["beta"],
            "log_beta_okx": btc_eth_eg["beta"],
            "ann_gross_okx": btc_eth_tr["ann_gross"],
            "ann_net_okx": btc_eth_tr["ann_net"],
            "n_trades_okx": int(btc_eth_tr["n_trades"]),
            "pnl_pvalue_dual": btc_eth_tr["pvalue_dual"],
            "pnl_pvalue_by": btc_eth_tr["pvalue_by"],
            "pnl_survives_rule5d": bool(btc_eth_tr["survives_rule5d_pnl"]),
        },
        "summary": {
            "n_pairs_eg_table_5pct": int(eg_df["significant_table_5pct"].sum()),
            "n_pairs_survive_eg_rule5d": n_coint_survive,
            "n_pairs_survive_pnl_rule5d": n_pnl_survive,
            "verdict": verdict,
        },
        "n_sims_eg": n_sims_eg,
        "n_sims_pnl": n_sims_pnl,
        "seed": seed,
        "note": (
            "Control only. Paper setup on OKX daily; Rule 5d dual-null+BY on EG (m=6) "
            "and traded PnL (m=4). interpretation_allowed=false."
        ),
    }

    if persist:
        config = config or load_config()
        out_dir = Path(config.storage.root) / eid / "cointegration"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "ijsra_rule5d_stress.json"
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        eg_df.to_parquet(out_dir / "ijsra_rule5d_eg.parquet", index=False)
        trade_df.to_parquet(out_dir / "ijsra_rule5d_trading.parquet", index=False)
        save_scan_results(
            eg_df,
            root=Path(config.storage.root),
            exchange_id=eid,
            timeframe=config.ohlcv.timeframe,
            mode="both",
            meta={"kind": "ijsra_rule5d_eg", "verdict": verdict, "n_sims_eg": n_sims_eg},
            basename="ijsra_rule5d_eg",
        )
        report["artifact"] = str(path)
        logger.info("IJSRA stress → %s verdict=%s", path, verdict)

    return report
