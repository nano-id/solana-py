"""Typed configuration for the data layer."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class ExchangeConfig(BaseModel):
    id: str = "binance"
    sandbox: bool = False
    enable_rate_limit: bool = True
    quote: str = "USDT"
    market_type: Literal["spot", "swap"] = "spot"


class UniverseConfig(BaseModel):
    top_n: int = Field(default=30, ge=2, le=200)
    min_quote_volume: float = Field(default=5_000_000.0, ge=0)
    exclude_bases: list[str] = Field(default_factory=list)

    @field_validator("exclude_bases")
    @classmethod
    def _upper_bases(cls, value: list[str]) -> list[str]:
        return [b.upper() for b in value]


class OhlcvConfig(BaseModel):
    timeframe: str = "1h"
    lookback_days: int = Field(default=365, ge=1)
    limit_per_request: int = Field(default=1000, ge=1, le=5000)


class StorageConfig(BaseModel):
    root: Path = Path("data")


class DataConfig(BaseModel):
    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    ohlcv: OhlcvConfig = Field(default_factory=OhlcvConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)


def load_config(path: str | Path | None = None) -> DataConfig:
    """Load data-layer config from YAML. Defaults if path is None/missing."""
    if path is None:
        candidates = [
            Path("config/default.yaml"),
            Path(__file__).resolve().parents[2] / "config" / "default.yaml",
        ]
        for candidate in candidates:
            if candidate.is_file():
                path = candidate
                break
        else:
            return DataConfig()

    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return DataConfig.model_validate(raw)
