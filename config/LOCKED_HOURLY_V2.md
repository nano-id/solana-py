# LOCKED — Retail hourly (1h) MM package

**Status:** FROZEN as of 2026-09-06. Do not recalibrate or overwrite.

## Canonical lock

- Params: `config/params_hourly_v2_jf2026.json`
- AS: `gamma=0.18`, `kappa=0.8`, `tau_bars=48`, `size_eta=2.5`, `adverse_vol_mult=0.45`
- Structure: `vol_lookback=192`, `ewma_halflife=96`, `liq_bps=6`, `vol_spread_mult=0.25`, `inventory_halt_frac=0.8`
- Sim: `fill_prob=0.25`, `adverse_frac=0.55`, `commission_pct=0.07`
- Config surface: `config/market_making.yaml` + retail `config/retail_100coin_*_v1.yaml`

## Rules

1. Do **not** run `spot-mm calibrate-hourly --write-config` against the locked YAMLs.
2. Do **not** overwrite `params_hourly_v2_jf2026.json` or `params_hourly_v1_archived.json`.
3. New timeframes (e.g. 15m) get their own modules, YAML, and `params_15m_*` locks.
4. Paper-only: `interpretation_allowed=false`.

Archived v1: `config/params_hourly_v1_archived.json`.
