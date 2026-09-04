"""Typed configuration for funding-rate arbitrage (Module 1)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from pairs_trading.data.config import ExchangeConfig, StorageConfig


class FundingOhlcvBridge(BaseModel):
    """Optional: reuse spot/swap OHLCV for basis checks later."""

    timeframe: str = "1h"
    lookback_days: int = Field(default=90, ge=1)


class FundingConfig(BaseModel):
    """Funding harvest settings. Thresholds are screens — not live entry params."""

    # History depth for fundingRateHistory pagination.
    lookback_days: int = Field(default=90, ge=7)
    limit_per_request: int = Field(default=100, ge=10, le=1000)
    # Typical crypto perp schedule: 3 settlements / day (8h). Override per venue if needed.
    settlements_per_day: float = Field(default=3.0, gt=0)
    # Absolute funding rate below this is treated as noise in screens (not a trade rule).
    min_abs_rate: float = Field(default=1e-5, ge=0)
    # Persist mode for screens.
    persist_mode: Literal["dated", "latest", "both"] = "both"
    # Default symbols if universe not used (CCXT swap symbols).
    default_symbols: list[str] = Field(
        default_factory=lambda: [
            "BTC/USDT:USDT",
            "ETH/USDT:USDT",
            "SOL/USDT:USDT",
            "XRP/USDT:USDT",
            "DOGE/USDT:USDT",
        ]
    )


class FundingAppConfig(BaseModel):
    exchange: ExchangeConfig = Field(
        default_factory=lambda: ExchangeConfig(
            id="binance",
            fallbacks=["bybit", "okx"],
            market_type="swap",
        )
    )
    storage: StorageConfig = Field(default_factory=StorageConfig)
    funding: FundingConfig = Field(default_factory=FundingConfig)
    ohlcv: FundingOhlcvBridge = Field(default_factory=FundingOhlcvBridge)

    @field_validator("exchange")
    @classmethod
    def _force_swap(cls, value: ExchangeConfig) -> ExchangeConfig:
        # Funding lives on perps; keep market_type=swap even if YAML says spot.
        value.market_type = "swap"
        return value


def load_funding_config(path: str | Path | None = None) -> FundingAppConfig:
    if path is None:
        candidates = [
            Path("config/funding.yaml"),
            Path(__file__).resolve().parents[2] / "config" / "funding.yaml",
        ]
        for candidate in candidates:
            if candidate.is_file():
                path = candidate
                break
        else:
            return FundingAppConfig()

    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return FundingAppConfig.model_validate(raw)
