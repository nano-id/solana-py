"""Typed configuration for spot market making."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class SimConfig(BaseModel):
    n_bars: int = Field(default=2000, ge=100)
    seed: int = 42
    mid0: float = Field(default=100.0, gt=0)
    ann_vol: float = Field(default=0.55, gt=0)
    fill_prob: float = Field(default=0.80, gt=0, le=1)
    adverse_frac: float = Field(default=0.15, ge=0, le=1)
    bars_per_year: float = Field(default=8760.0, gt=0)


class MarketMakingConfig(BaseModel):
    """Spot MM. commission_pct is percent points: 0.07 → 0.07% = 7 bps/side."""

    symbols: list[str] = Field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
    commission_pct: float = Field(default=0.07, ge=0)
    edge_buffer_bps: float = Field(default=5.0, ge=0)
    min_half_spread_bps: float = Field(default=12.0, gt=0)
    inventory_skew_bps: float = Field(default=8.0, ge=0)
    target_inventory: float = 0.0
    max_inventory: float = Field(default=1.0, gt=0)
    size_frac_of_max: float = Field(default=0.1, gt=0, le=1)
    inventory_halt_frac: float = Field(default=0.95, gt=0, le=1)
    vol_lookback: int = Field(default=48, ge=5)
    vol_spread_mult: float = Field(default=0.35, ge=0)
    sim: SimConfig = Field(default_factory=SimConfig)
    interpretation_allowed: bool = False
    storage_root: str = "data"
    exchange_id: str = "okx"

    @field_validator("commission_pct")
    @classmethod
    def _reject_fraction_typo(cls, v: float) -> float:
        if 0 < v < 0.001:
            raise ValueError(
                "commission_pct looks like a fraction; pass percent points "
                "(0.07 for 0.07%, not 0.0007)."
            )
        return v

    @property
    def commission_frac(self) -> float:
        return self.commission_pct / 100.0

    @property
    def commission_bps(self) -> float:
        return self.commission_pct * 100.0

    @property
    def required_half_spread_bps(self) -> float:
        return max(self.min_half_spread_bps, self.commission_bps + self.edge_buffer_bps)


def load_mm_config(path: str | Path | None = None) -> MarketMakingConfig:
    path = Path(path or "config/market_making.yaml")
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    mm = dict(raw.get("market_making") or {})
    storage = raw.get("storage") or {}
    exchange = raw.get("exchange") or {}
    if "root" in storage:
        mm.setdefault("storage_root", storage["root"])
    if "id" in exchange:
        mm.setdefault("exchange_id", exchange["id"])
    return MarketMakingConfig.model_validate(mm)
