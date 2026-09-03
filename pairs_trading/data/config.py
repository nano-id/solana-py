"""Typed configuration for the data layer."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class ExchangeConfig(BaseModel):
    """Exchange selection is config-driven — never hard-coded at call sites."""

    id: str = "binance"
    # Tried in order after `id` on geo/availability failures (403/451/etc.).
    fallbacks: list[str] = Field(default_factory=lambda: ["bybit", "okx"])
    sandbox: bool = False
    enable_rate_limit: bool = True
    quote: str = "USDT"
    market_type: Literal["spot", "swap"] = "spot"

    @field_validator("id")
    @classmethod
    def _lower_id(cls, value: str) -> str:
        return value.lower().strip()

    @field_validator("fallbacks")
    @classmethod
    def _lower_fallbacks(cls, value: list[str]) -> list[str]:
        return [v.lower().strip() for v in value]

    def candidate_ids(self) -> list[str]:
        """Primary exchange first, then unique fallbacks."""
        ordered = [self.id, *[f for f in self.fallbacks if f != self.id]]
        # Preserve order, drop duplicates.
        seen: set[str] = set()
        out: list[str] = []
        for name in ordered:
            if name not in seen:
                seen.add(name)
                out.append(name)
        return out


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


class CointegrationConfig(BaseModel):
    """Scan settings. Significance threshold is a statistical convention (α),
    not a strategy entry parameter — strategy thresholds belong in walk-forward."""

    pvalue_threshold: float = Field(default=0.05, gt=0, lt=1)
    min_obs: int = Field(default=100, ge=30)
    # How to persist scan outputs: dated folder keeps history; latest overwrites.
    persist_mode: Literal["dated", "latest", "both"] = "both"
    # Point-in-time universe: use snapshot dated on/before this (YYYY-MM-DD),
    # or "auto" = earliest overlapping bar date across the series.
    universe_as_of: str = "auto"
    # log => log(y)=α+β·log(x)+ε (default, pairs-trading standard).
    # level => raw prices (only for diagnostics; do not use for Phase 3 signals).
    price_space: Literal["log", "level"] = "log"


class DataConfig(BaseModel):
    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    ohlcv: OhlcvConfig = Field(default_factory=OhlcvConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    cointegration: CointegrationConfig = Field(default_factory=CointegrationConfig)


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
