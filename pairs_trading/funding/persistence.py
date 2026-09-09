"""Funding-rate sign persistence / autocorrelation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def funding_persistence(rates: pd.Series, *, lag: int = 1) -> dict[str, float]:
    s = rates.astype(float).dropna()
    nan = float("nan")
    if len(s) < lag + 5:
        return {
            "sign_persistence": nan,
            "autocorr_lag1": nan,
            "same_sign_streak_mean": nan,
            "harvest_efficiency": nan,
        }

    signs = np.sign(s.to_numpy())
    a, b = signs[:-lag], signs[lag:]
    both_nz = (a != 0) & (b != 0)
    sign_pers = float((a[both_nz] == b[both_nz]).mean()) if both_nz.sum() else nan

    x = s.to_numpy()
    if np.std(x[:-lag]) < 1e-15 or np.std(x[lag:]) < 1e-15:
        ac = nan
    else:
        ac = float(np.corrcoef(x[:-lag], x[lag:])[0, 1])

    streaks: list[int] = []
    cur = 1
    for i in range(1, len(signs)):
        if signs[i] == signs[i - 1] and signs[i] != 0:
            cur += 1
        else:
            if signs[i - 1] != 0:
                streaks.append(cur)
            cur = 1
    if signs[-1] != 0:
        streaks.append(cur)
    streak_mean = float(np.mean(streaks)) if streaks else nan

    mean_abs = float(np.mean(np.abs(x)))
    eff = nan if mean_abs < 1e-15 else float(abs(float(np.mean(x))) / mean_abs)

    return {
        "sign_persistence": sign_pers,
        "autocorr_lag1": ac,
        "same_sign_streak_mean": streak_mean,
        "harvest_efficiency": eff,
    }
