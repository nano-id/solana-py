"""Factor-neutral preprocessing: regress log(coin) on factor(s), keep residual."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)

# Majors excluded from the equal-weight altcoin index.
DEFAULT_ALT_INDEX_EXCLUDE = ("BTC/USDT", "ETH/USDT", "BNB/USDT")


def _log_close(ohlcv: pd.DataFrame) -> pd.DataFrame:
    out = ohlcv[["timestamp", "close"]].copy()
    close = out["close"].astype(float)
    if (close <= 0).any():
        raise ValueError("Non-positive close cannot be log-transformed")
    out["log_close"] = np.log(close)
    return out[["timestamp", "log_close"]]


def residualize_against_factor(
    coin_ohlcv: pd.DataFrame,
    factor_ohlcv: pd.DataFrame,
    *,
    min_obs: int,
) -> tuple[pd.DataFrame, float, float, int]:
    """OLS: log(coin) = α + β·log(factor) + ε. Return residual as OHLCV-like frame.

    The returned frame uses `close` = residual (already in log-space). Callers must
    NOT log-transform again (use price_space='level' in Engle-Granger).
    """
    left = _log_close(coin_ohlcv).rename(columns={"log_close": "y"})
    right = _log_close(factor_ohlcv).rename(columns={"log_close": "f"})
    merged = left.merge(right, on="timestamp", how="inner").sort_values("timestamp")
    merged = merged.dropna()
    if len(merged) < min_obs:
        raise ValueError(
            f"Insufficient overlap for residualization: {len(merged)} < {min_obs}"
        )

    x = sm.add_constant(merged["f"].to_numpy(), has_constant="add")
    model = sm.OLS(merged["y"].to_numpy(), x).fit()
    alpha = float(model.params[0])
    beta = float(model.params[1])
    resid = pd.Series(model.resid, index=merged.index)

    frame = pd.DataFrame(
        {
            "timestamp": merged["timestamp"].to_numpy(),
            "open": resid.to_numpy(),
            "high": resid.to_numpy(),
            "low": resid.to_numpy(),
            "close": resid.to_numpy(),
            "volume": np.ones(len(resid)),
        }
    )
    return frame, alpha, beta, len(frame)


def residualize_against_factors(
    coin_ohlcv: pd.DataFrame,
    factor_frames: dict[str, pd.DataFrame],
    *,
    min_obs: int,
) -> tuple[pd.DataFrame, dict[str, float], int]:
    """OLS: log(coin) = α + Σ β_k · F_k + ε.

    Factor frames use timestamp + close (log-price or cumsum of EW log-returns).
    """
    if not factor_frames:
        raise ValueError("factor_frames is empty")
    left = _log_close(coin_ohlcv).rename(columns={"log_close": "y"})
    merged = left
    names: list[str] = []
    for name, fdf in factor_frames.items():
        col = f"f_{name}"
        piece = fdf[["timestamp", "close"]].rename(columns={"close": col})
        piece[col] = piece[col].astype(float)
        merged = merged.merge(piece, on="timestamp", how="inner")
        names.append(name)
    merged = merged.dropna().sort_values("timestamp")
    if len(merged) < min_obs:
        raise ValueError(
            f"Insufficient overlap for multi-factor residualization: "
            f"{len(merged)} < {min_obs}"
        )
    x_cols = [f"f_{n}" for n in names]
    x = sm.add_constant(merged[x_cols].to_numpy(), has_constant="add")
    model = sm.OLS(merged["y"].to_numpy(), x).fit()
    coeffs = {"alpha": float(model.params[0])}
    for i, name in enumerate(names):
        coeffs[f"beta_{name}"] = float(model.params[i + 1])
    resid = model.resid
    frame = pd.DataFrame(
        {
            "timestamp": merged["timestamp"].to_numpy(),
            "open": resid,
            "high": resid,
            "low": resid,
            "close": resid,
            "volume": np.ones(len(resid)),
        }
    )
    return frame, coeffs, len(frame)


def build_equal_weight_alt_index(
    frames: dict[str, pd.DataFrame],
    *,
    exclude: tuple[str, ...] = DEFAULT_ALT_INDEX_EXCLUDE,
    min_members: int = 3,
) -> tuple[pd.DataFrame, list[str]]:
    """Equal-weight average of daily log-returns → cumulative log-index.

    Members: every symbol in ``frames`` whose name is not in ``exclude``.
    Index level starts at 0; ``close`` is the cumulative sum of mean Δlog.
    """
    members = sorted(s for s in frames if s not in exclude)
    if len(members) < min_members:
        raise ValueError(
            f"Need ≥{min_members} alt members for index; have {members}"
        )

    log_panels: list[pd.DataFrame] = []
    for sym in members:
        log_panels.append(_log_close(frames[sym]).rename(columns={"log_close": sym}))

    panel = log_panels[0]
    for p in log_panels[1:]:
        panel = panel.merge(p, on="timestamp", how="inner")
    panel = panel.sort_values("timestamp").dropna()
    if len(panel) < 3:
        raise ValueError("Alt index panel too short after inner-join")

    rets = panel[members].diff().iloc[1:]
    ts = panel["timestamp"].iloc[1:]
    eq = rets.mean(axis=1)
    index_level = eq.cumsum()
    frame = pd.DataFrame(
        {
            "timestamp": ts.to_numpy(),
            "open": index_level.to_numpy(),
            "high": index_level.to_numpy(),
            "low": index_level.to_numpy(),
            "close": index_level.to_numpy(),
            "volume": np.ones(len(index_level)),
        }
    )
    logger.info(
        "Alt index: %d members, %d bars (excl %s)",
        len(members),
        len(frame),
        ",".join(exclude),
    )
    return frame, members


def build_factor_residual_frames(
    frames: dict[str, pd.DataFrame],
    *,
    factor_symbol: str,
    min_obs: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, float | int]]]:
    """Residualize every non-factor symbol against `factor_symbol`."""
    if factor_symbol not in frames:
        raise FileNotFoundError(
            f"Factor symbol {factor_symbol} missing from loaded frames"
        )
    factor = frames[factor_symbol]
    out: dict[str, pd.DataFrame] = {}
    meta: dict[str, dict[str, float | int]] = {}
    for sym, ohlcv in frames.items():
        if sym == factor_symbol:
            continue
        try:
            resid, alpha, beta, n = residualize_against_factor(
                ohlcv, factor, min_obs=min_obs
            )
        except ValueError as exc:
            logger.warning("Skip residualizing %s: %s", sym, exc)
            continue
        out[sym] = resid
        meta[sym] = {"alpha": alpha, "beta_vs_factor": beta, "n_obs": n}
        logger.info(
            "Residualized %s vs %s: β_factor=%.4f n=%d",
            sym,
            factor_symbol,
            beta,
            n,
        )
    return out, meta


def build_btc_alt_residual_frames(
    frames: dict[str, pd.DataFrame],
    *,
    btc_symbol: str = "BTC/USDT",
    exclude_from_alt_index: tuple[str, ...] = DEFAULT_ALT_INDEX_EXCLUDE,
    min_obs: int,
    symbols: list[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict], pd.DataFrame, list[str]]:
    """Residualize coins vs BTC log-price + equal-weight alt index.

    Returns (residual_frames, meta_by_symbol, alt_index_frame, alt_members).
    """
    if btc_symbol not in frames:
        raise FileNotFoundError(f"BTC factor {btc_symbol} missing")
    alt_index, members = build_equal_weight_alt_index(
        frames, exclude=exclude_from_alt_index
    )
    targets = symbols if symbols is not None else [
        s for s in frames if s != btc_symbol
    ]
    factor_frames = {"btc": frames[btc_symbol], "alt_index": alt_index}
    out: dict[str, pd.DataFrame] = {}
    meta: dict[str, dict] = {}
    for sym in targets:
        if sym not in frames:
            continue
        try:
            resid, coeffs, n = residualize_against_factors(
                frames[sym], factor_frames, min_obs=min_obs
            )
        except ValueError as exc:
            logger.warning("Skip BTC+alt residual %s: %s", sym, exc)
            continue
        out[sym] = resid
        meta[sym] = {**coeffs, "n_obs": n}
        logger.info(
            "Residualized %s vs BTC+alt: β_btc=%.4f β_alt=%.4f n=%d",
            sym,
            coeffs.get("beta_btc", float("nan")),
            coeffs.get("beta_alt_index", float("nan")),
            n,
        )
    return out, meta, alt_index, members
