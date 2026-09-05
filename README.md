# pairs-trading

Kripto piyasaları için istatistiksel / mekanik arbitraj altyapısı.

**Öncelik (2026-09):** Modül 1 — perpetual funding-rate harvest (delta-nötr).  
Modül 2 (kointegrasyon tarama) araştırma olarak **kapatıldı** — kod referans olarak duruyor; bu venue/dönemde çoklu-test + bağımlılık-sağlam kointegre çift bulunamadı.

## Kurulum

```bash
pip install -e ".[dev]"
```

## Modül 1 — Funding rate (aktif)

Perp funding geçmişi → net carry harvest (fees + persistence + basis + capacity + sizing) → paper backtest.

```bash
pairs-funding --config config/funding.yaml sync
pairs-funding --config config/funding.yaml screen
pairs-funding --config config/funding.yaml harvest --persist
pairs-funding --config config/funding.yaml paper --persist
pairs-funding show BTC/USDT:USDT
```

```
data/{exchange}/funding/{SYMBOL}.parquet
data/{exchange}/funding/screens/harvest_latest.{parquet,csv,meta.json}
data/{exchange}/funding/screens/paper_latest.{parquet,csv,meta.json}
```

`exchange.market_type` kodda **swap** zorlanır. Geo-fallback: binance → bybit → okx.

**Canlı OKX smoke (2026-09):** harvest `positive_net≥3` ve `tradable≥3` (DOGE/NEAR/SUI); paper `18/18` pozitif net PnL (~90g funding history, fee dahil).

## Faz 1 — OHLCV veri katmanı

```bash
pairs-data --config config/smoke.yaml sync --symbols BTC/USDT ETH/USDT --no-universe
pairs-data universe --print
pairs-data show BTC/USDT
```

## Modül 2 — Kointegrasyon (araştırma, kapalı)

Engle-Granger / Johansen, BH/BY-FDR, hub/placebo, residualization, dual-null (RW + block-bootstrap). Sonuç: OKX ~2022–2026’da sağlam survivor yok. Kod silinmez — ileride faktör-bazlı hat için referans.

```bash
pairs-coint scan --config config/default.yaml
pairs-coint majors --config config/majors.yaml
```

## Test

```bash
pytest -q                 # unit (network yok)
pytest -q -m integration  # canlı exchange smoke
```

## Yol haritası (Modül 1)

1. ~~Funding history sync + screen~~
2. ~~Basis / cash-and-carry spread ekranı (perp vs spot)~~
3. ~~Carry net of fees + funding persistence~~
4. ~~Kapasite / likidite filtresi~~
5. ~~Risk / pozisyon boyutlandırma (delta-nötr)~~
6. ~~Paper harvest backtest + screen persist~~
7. Live execution / walk-forward (sonraki dilim)

## Tasarım ilkeleri

- Strateji eşikleri walk-forward / risk katmanından gelir; screen ≠ entry.
- Look-ahead: yalnızca kapanmış settlement / mum.
- Survivorship: universe tarihli snapshot (OHLCV tarafı).
