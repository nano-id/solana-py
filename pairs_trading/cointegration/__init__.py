"""Cointegration scanning: Engle-Granger, hedge ratio, half-life."""

from pairs_trading.cointegration.engine import (
    CointegrationHit,
    engle_granger_pair,
    evaluate_pair,
)
from pairs_trading.cointegration.half_life import estimate_half_life
from pairs_trading.cointegration.persist import save_scan_results
from pairs_trading.cointegration.scanner import scan_pairs

__all__ = [
    "CointegrationHit",
    "estimate_half_life",
    "evaluate_pair",
    "engle_granger_pair",
    "save_scan_results",
    "scan_pairs",
]
