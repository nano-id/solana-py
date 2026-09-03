"""Cointegration scanning: Engle-Granger, hedge ratio, half-life."""

from pairs_trading.cointegration.scanner import (
    CointegrationHit,
    scan_pairs,
    engle_granger_pair,
)
from pairs_trading.cointegration.half_life import estimate_half_life
from pairs_trading.cointegration.persist import save_scan_results

__all__ = [
    "CointegrationHit",
    "estimate_half_life",
    "save_scan_results",
    "scan_pairs",
    "engle_granger_pair",
]
