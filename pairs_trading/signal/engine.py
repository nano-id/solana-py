"""Z-score entry/exit/stop signal engine (mechanical skeleton)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd


class Side(str, Enum):
    FLAT = "flat"
    LONG_SPREAD = "long_spread"
    SHORT_SPREAD = "short_spread"


@dataclass(frozen=True)
class SignalConfig:
    """Mechanical thresholds for skeleton verification only.

    Walk-forward must replace these before any live risk is taken.
    """

    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    z_window: int = 60


@dataclass(frozen=True)
class SignalEvent:
    bar: int
    side: Side
    z: float
    reason: Literal["enter_long", "enter_short", "exit_mean", "exit_stop", "hold"]


def rolling_zscore(spread: pd.Series, window: int) -> pd.Series:
    mu = spread.rolling(window, min_periods=window).mean()
    sd = spread.rolling(window, min_periods=window).std(ddof=0)
    z = (spread - mu) / sd.replace(0.0, np.nan)
    return z.rename("z")


def generate_signals(
    spread: pd.Series,
    config: SignalConfig | None = None,
) -> tuple[pd.DataFrame, list[SignalEvent]]:
    """Map spread → z-score → long/short/flat with mean-exit and stop-out."""
    config = config or SignalConfig()
    z = rolling_zscore(spread, config.z_window)
    side = Side.FLAT
    events: list[SignalEvent] = []
    sides: list[str] = []
    reasons: list[str] = []

    for i in range(len(spread)):
        zi = z.iloc[i]
        reason: Literal["enter_long", "enter_short", "exit_mean", "exit_stop", "hold"] = "hold"
        if not np.isnan(zi):
            if side == Side.FLAT:
                if zi <= -config.entry_z:
                    side = Side.LONG_SPREAD
                    reason = "enter_long"
                elif zi >= config.entry_z:
                    side = Side.SHORT_SPREAD
                    reason = "enter_short"
            else:
                if abs(zi) >= config.stop_z:
                    side = Side.FLAT
                    reason = "exit_stop"
                elif abs(zi) <= config.exit_z:
                    side = Side.FLAT
                    reason = "exit_mean"
            if reason != "hold":
                events.append(SignalEvent(bar=i, side=side, z=float(zi), reason=reason))
        sides.append(side.value)
        reasons.append(reason)

    frame = pd.DataFrame(
        {
            "spread": spread.to_numpy(),
            "z": z.to_numpy(),
            "side": sides,
            "reason": reasons,
        }
    )
    return frame, events


def summarize_signals(frame: pd.DataFrame, events: list[SignalEvent]) -> dict:
    return {
        "n_bars": int(len(frame)),
        "n_events": len(events),
        "n_entries": sum(1 for e in events if e.reason.startswith("enter_")),
        "n_exit_mean": sum(1 for e in events if e.reason == "exit_mean"),
        "n_exit_stop": sum(1 for e in events if e.reason == "exit_stop"),
        "bars_in_position": int((frame["side"] != Side.FLAT.value).sum()),
    }
