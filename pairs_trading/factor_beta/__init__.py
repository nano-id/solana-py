"""Module 3 — BTC-factor residual strategy (factor beta)."""

from pairs_trading.factor_beta.config import FactorBetaConfig, load_factor_beta_config
from pairs_trading.factor_beta.data import LogPricePanel, build_log_price_panel, inventory_report

__all__ = [
    "FactorBetaConfig",
    "LogPricePanel",
    "build_log_price_panel",
    "inventory_report",
    "load_factor_beta_config",
]
