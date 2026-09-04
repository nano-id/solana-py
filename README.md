# pairs-trading

Kripto piyasaları için istatistiksel / mekanik arbitraj altyapısı.

**Öncelik (2026-09):** Modül 1 — perpetual funding-rate harvest (delta-nötr).  
Modül 2 (kointegrasyon tarama) araştırma olarak **kapatıldı** — kod referans olarak duruyor; bu venue/dönemde çoklu-test + bağımlılık-sağlam kointegre çift bulunamadı.

## Kurulum

```bash
pip install -e ".[dev]"
```

## Modül 1 — Funding rate (aktif)

Perp funding geçmişini çeker, parquet saklar, |yıllıklandırılmış ortalama| ile harvest adaylarını sıralar. Eşikler ekran içindir — canlı entry kuralı değil.

```bash
pairs-funding --config config/funding.yaml sync
pairs-funding --config config/funding.yaml sync --symbols BTC/USDT:USDT ETH/USDT:USDT
pairs-funding show BTC/USDT:USDT
pairs-funding screen
```

```
data/{exchange}/funding/{SYMBOL}.parquet
```

`exchange.market_type` kodda **swap** zorlanır. Geo-fallback: binance → bybit → okx (Faz 1 ile aynı).

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

1. ~~Funding history sync + screen~~ (bu PR)
2. Basis / cash-and-carry spread ekranı (perp vs spot)
3. Carry net of fees + funding persistence
4. Kapasite / likidite filtresi
5. Risk / pozisyon boyutlandırma (delta-nötr)
6. Paper → live execution (mevcut execution iskeletiyle)

## Tasarım ilkeleri

- Strateji eşikleri walk-forward / risk katmanından gelir; screen ≠ entry.
- Look-ahead: yalnızca kapanmış settlement / mum.
- Survivorship: universe tarihli snapshot (OHLCV tarafı).
