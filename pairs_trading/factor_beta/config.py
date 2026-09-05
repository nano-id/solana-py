"""Module 3 factor-beta config. Rule-5 fields are required and validated."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class FactorConfig(BaseModel):
    symbol: str = "BTC/USDT"
    beta_window_grid: list[int] = Field(default_factory=lambda: [60, 90, 120])
    min_obs: int = Field(default=60, ge=30)


class UniverseConfig(BaseModel):
    symbols: list[str]
    require_on_disk: bool = True
    prefer_on_disk_intersection: bool = True
    max_symbols: int = Field(default=20, ge=2, le=50)


class CostsConfig(BaseModel):
    round_trip_bps_min: float = Field(default=30.0, gt=0)
    round_trip_bps_max: float = Field(default=40.0, gt=0)
    round_trip_bps: float = Field(default=35.0, gt=0)
    source: str = "OKX Regular dual-leg taker RT"

    @model_validator(mode="after")
    def _ok(self) -> CostsConfig:
        if self.round_trip_bps_min > self.round_trip_bps_max:
            raise ValueError("round_trip_bps_min > max")
        if not (self.round_trip_bps_min <= self.round_trip_bps <= self.round_trip_bps_max):
            raise ValueError("round_trip_bps outside [min, max]")
        if self.round_trip_bps < 30.0:
            raise ValueError("Rule 5a: round_trip_bps must be ≥ 30 (5 bps forbidden)")
        return self


class EntryGateConfig(BaseModel):
    min_edge_to_cost_multiple: float = Field(default=2.5, ge=2.5)
    max_edge_to_cost_multiple: float = Field(default=3.0, ge=2.5)

    @model_validator(mode="after")
    def _ok(self) -> EntryGateConfig:
        if self.min_edge_to_cost_multiple > self.max_edge_to_cost_multiple:
            raise ValueError("min_edge_to_cost_multiple > max")
        return self


class EvaluationConfig(BaseModel):
    min_independent_windows: int = Field(default=5, ge=5)
    window_weeks_min: int = Field(default=8, ge=8)
    window_weeks_max: int = Field(default=10, ge=8)
    windows_must_be_non_overlapping: bool = True

    @model_validator(mode="after")
    def _ok(self) -> EvaluationConfig:
        if not self.windows_must_be_non_overlapping:
            raise ValueError("Rule 5c: windows must be non-overlapping")
        if self.window_weeks_min > self.window_weeks_max:
            raise ValueError("window_weeks_min > max")
        return self


class MultipleTestingConfig(BaseModel):
    dual_null: bool = True
    null_a: Literal["independent_random_walk"] = "independent_random_walk"
    null_b: Literal["block_bootstrap"] = "block_bootstrap"
    fdr_method: Literal["benjamini_yekutieli"] = "benjamini_yekutieli"
    fdr_q: float = Field(default=0.10, gt=0, lt=1)

    @model_validator(mode="after")
    def _ok(self) -> MultipleTestingConfig:
        if not self.dual_null:
            raise ValueError("Rule 5d: dual_null must be true")
        return self


class CommonFactorControlConfig(BaseModel):
    max_abs_residual_corr: float = Field(default=0.5, gt=0, lt=1)
    action: Literal["cluster_as_single_signal"] = "cluster_as_single_signal"


class StationarityConfig(BaseModel):
    adf_pvalue_max: float = Field(default=0.05, gt=0, lt=1)


class SignalConfig(BaseModel):
    z_entry_grid: list[float] = Field(default_factory=lambda: [1.5, 2.0, 2.5, 3.0])
    z_exit_grid: list[float] = Field(default_factory=lambda: [0.25, 0.5, 0.75])
    zscore_window_grid: list[int] = Field(default_factory=lambda: [20, 40, 60])


class WalkForwardConfig(BaseModel):
    prefer_plateau: bool = True
    plateau_neighbor_tolerance_pct: float = Field(default=10.0, gt=0)


class LoggingConfig(BaseModel):
    report_gross_and_net_separately: bool = True
    report_trade_count: bool = True
    report_cost_leakage_ratio: bool = True


class StorageConfig(BaseModel):
    root: Path = Path("data")
    exchange_id: str = "okx"
    timeframe: str = "1d"
    output_dir: Path = Path("data/okx/factor_beta")


class FactorBetaConfig(BaseModel):
    data_config: Path = Path("config/scan.yaml")
    factor: FactorConfig = Field(default_factory=FactorConfig)
    universe: UniverseConfig
    costs: CostsConfig = Field(default_factory=CostsConfig)
    entry_gate: EntryGateConfig = Field(default_factory=EntryGateConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    multiple_testing: MultipleTestingConfig = Field(default_factory=MultipleTestingConfig)
    common_factor_control: CommonFactorControlConfig = Field(
        default_factory=CommonFactorControlConfig
    )
    stationarity: StationarityConfig = Field(default_factory=StationarityConfig)
    signal: SignalConfig = Field(default_factory=SignalConfig)
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)


def load_factor_beta_config(path: str | Path | None = None) -> FactorBetaConfig:
    if path is None:
        candidates = [
            Path("config/factor_beta.yaml"),
            Path(__file__).resolve().parents[2] / "config" / "factor_beta.yaml",
        ]
        for c in candidates:
            if c.is_file():
                path = c
                break
        else:
            raise FileNotFoundError("config/factor_beta.yaml not found")
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return FactorBetaConfig.model_validate(raw)
