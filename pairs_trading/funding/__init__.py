"""Module 1 — perpetual funding-rate arbitrage."""

from pairs_trading.funding.config import FundingAppConfig, load_funding_config
from pairs_trading.funding.pipeline import FundingPipeline
from pairs_trading.funding.screen import annualize_funding_rate, screen_funding_store

__all__ = [
    "FundingAppConfig",
    "FundingPipeline",
    "annualize_funding_rate",
    "load_funding_config",
    "screen_funding_store",
]
