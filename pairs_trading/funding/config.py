"""Typed configuration for funding-rate arbitrage (Module 1)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from pairs_trading.data.config import ExchangeConfig, StorageConfig
from pairs_trading.funding.capacity import CapacityConfig
from pairs_trading.funding.fees import FeeConfig
from pairs_trading.funding.risk import RiskConfig


class FundingOhlcvBridge(BaseModel):
    timeframe: str = "1h"
    lookback_days: int = Field(default=90, ge=1)


class HarvestConfig(BaseModel):
    """Net-carry screen knobs (not live entry rules)."""

    expected_hold_days: float = Field(default=30.0, gt=0)
    min_net_ann: float = Field(default=0.0)
    apply_persistence: bool = True
    apply_basis_drag: bool = True


class FundingConfig(BaseModel):
    """Funding harvest settings. Thresholds are screens — not live entry params."""

    lookback_days: int = Field(default=90, ge=7)
    limit_per_request: int = Field(default=100, ge=10, le=1000)
    settlements_per_day: float = Field(default=3.0, gt=0)
    min_abs_rate: float = Field(default=1e-5, ge=0)
    persist_mode: Literal["dated", "latest", "both"] = "both"
    default_symbols: list[str] = Field(
        default_factory=lambda: [
            "BTC/USDT:USDT",
            "ETH/USDT:USDT",
            "SOL/USDT:USDT",
            "XRP/USDT:USDT",
            "DOGE/USDT:USDT",
            "LTC/USDT:USDT",
            "ADA/USDT:USDT",
            "LINK/USDT:USDT",
            "AVAX/USDT:USDT",
            "SUI/USDT:USDT",
            "APT/USDT:USDT",
            "NEAR/USDT:USDT",
            "ARB/USDT:USDT",
            "OP/USDT:USDT",
            "INJ/USDT:USDT",
            "FIL/USDT:USDT",
            "WLD/USDT:USDT",
            "ORDI/USDT:USDT",
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
    fees: FeeConfig = Field(default_factory=FeeConfig)
    capacity: CapacityConfig = Field(default_factory=CapacityConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    harvest: HarvestConfig = Field(default_factory=HarvestConfig)

    @field_validator("exchange")
    @classmethod
    def _force_swap(cls, value: ExchangeConfig) -> ExchangeConfig:
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
