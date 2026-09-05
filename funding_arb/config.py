"""Typed configuration for the funding_arb collector (step 1)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from pairs_trading.data.config import ExchangeConfig, StorageConfig


# Top ~20 liquid USDT-m perps — shared with Module 1 screen universe.
_DEFAULT_SYMBOLS: list[str] = [
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
    "BNB/USDT:USDT",
    "DOT/USDT:USDT",
]


class DepthConfig(BaseModel):
    """Order-book slice used for slippage / capacity estimates."""

    book_limit: int = Field(default=20, ge=5, le=200)
    # Sum notional within this fraction of mid (e.g. 0.002 = 20 bps).
    band_frac: float = Field(default=0.002, gt=0, le=0.05)
    fetch_spot_book: bool = True
    fetch_perp_book: bool = True


class CollectorConfig(BaseModel):
    """Periodic live snapshot loop."""

    interval_seconds: int = Field(default=900, ge=30)  # 15 min
    symbols: list[str] = Field(default_factory=lambda: list(_DEFAULT_SYMBOLS))
    persist_mode: Literal["dated", "latest", "both"] = "both"
    settlements_per_day: float = Field(default=3.0, gt=0)  # 8h funding cycle


class ScreenConfig(BaseModel):
    """Step 2 opportunity screen knobs (not live entry rules)."""

    min_depth_usdt: float = Field(default=10_000.0, ge=0)
    # Primary rate column: next settlement (funding_rate). sett is tie-break only.
    rate_col: str = "funding_rate"
    # How many recent snapshots to score interest-floor stickiness.
    floor_lookback: int = Field(default=12, ge=1, le=500)
    # Rank real premium (non-floor) above structural interest-floor names.
    prefer_non_floor: bool = True


class EntryConfig(BaseModel):
    """Step 3 paper / small-notional delta-neutral entry knobs."""

    depth_fraction: float = Field(default=0.05, gt=0, le=1.0)
    max_notional_usdt: float = Field(default=100.0, gt=0)
    min_notional_usdt: float = Field(default=10.0, ge=0)
    require_non_floor: bool = True
    flatten_unpaired: bool = True
    # Paper mode only until live wiring is added.
    mode: Literal["paper", "live"] = "paper"


class RotationConfig(BaseModel):
    """Step 4 rotation gate (non-floor premium multiple)."""

    min_edge_multiple: float = Field(default=3.0, gt=1.0)
    # Assumed round-trip cost fraction for gross-vs-net honesty (not a trigger).
    rotation_cost: float = Field(default=0.004, ge=0.0, le=0.05)
    require_non_floor: bool = True


class ExitConfig(BaseModel):
    """Step 5 exit triggers."""

    exit_on_adverse_funding: bool = True
    exit_on_floor: bool = True


class JournalConfig(BaseModel):
    """Step 6 trade journal + paper gross/net cost assumptions."""

    enabled: bool = True
    # Fraction of notional charged on rotate (informational netting).
    rotation_cost: float = Field(default=0.004, ge=0.0, le=0.05)
    # Fraction of notional charged on each open or close event.
    slippage_cost: float = Field(default=0.0005, ge=0.0, le=0.05)


class ObservationConfig(BaseModel):
    """Multi-day collect readiness gates (NOT a coding-complete flag)."""

    min_days: float = Field(default=5.0, gt=0)
    min_settlements: int = Field(default=15, ge=1)
    floor_lookback: int = Field(default=48, ge=1)


class FundingArbAppConfig(BaseModel):
    exchange: ExchangeConfig = Field(
        default_factory=lambda: ExchangeConfig(
            id="binance",
            fallbacks=["bybit", "okx"],
            market_type="swap",
        )
    )
    storage: StorageConfig = Field(default_factory=StorageConfig)
    collector: CollectorConfig = Field(default_factory=CollectorConfig)
    depth: DepthConfig = Field(default_factory=DepthConfig)
    screen: ScreenConfig = Field(default_factory=ScreenConfig)
    entry: EntryConfig = Field(default_factory=EntryConfig)
    rotation: RotationConfig = Field(default_factory=RotationConfig)
    exit: ExitConfig = Field(default_factory=ExitConfig)
    journal: JournalConfig = Field(default_factory=JournalConfig)
    observation: ObservationConfig = Field(default_factory=ObservationConfig)

    @field_validator("exchange")
    @classmethod
    def _force_swap(cls, value: ExchangeConfig) -> ExchangeConfig:
        value.market_type = "swap"
        return value


def load_funding_arb_config(path: str | Path | None = None) -> FundingArbAppConfig:
    if path is None:
        candidates = [
            Path("config/funding_arb.yaml"),
            Path(__file__).resolve().parents[1] / "config" / "funding_arb.yaml",
        ]
        for candidate in candidates:
            if candidate.is_file():
                path = candidate
                break
        else:
            return FundingArbAppConfig()

    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return FundingArbAppConfig.model_validate(raw)
