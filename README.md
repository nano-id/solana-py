# pairs-trading

Kointegrasyon bazlı istatistiksel arbitraj (pairs trading) sistemi — kripto piyasaları.

## Kurulum

```bash
pip install -e ".[dev]"
```

## Faz 1 — Veri katmanı

CCXT ile config’den seçilen borsadan (`binance` | `bybit` | `okx`) üst coin evrenini seçer, OHLCV çeker, parquet saklar.

Borsa seçimi **config-driven**. `exchange.id` varsayılan `binance`; `exchange.fallbacks` (örn. `[bybit, okx]`) ile 403/451 gibi bölgesel hatalarda otomatik sırayla denenir. Hangi borsadan veri geldiği loglanır.

```bash
# --config sıradan bağımsız:
pairs-data --config config/smoke.yaml sync --symbols BTC/USDT ETH/USDT --no-universe
pairs-data sync --config config/smoke.yaml --symbols BTC/USDT ETH/USDT --no-universe

pairs-data universe --print
pairs-data show BTC/USDT
```

```
data/{exchange}/universe/{YYYY-MM-DD}.json
data/{exchange}/{timeframe}/{SYMBOL}.parquet
```

## Faz 2 — Kointegrasyon tarama

Engle-Granger (`statsmodels.coint`), OLS hedge oranı (β), AR(1) half-life. Universe point-in-time snapshot’tan alınır.

```bash
# Tek çift (sync + test)
pairs-coint pair ETH/USDT BTC/USDT --sync --config config/smoke.yaml

# Evren taraması (tarihli + latest sonuç dosyası)
pairs-coint scan --config config/default.yaml
pairs-coint scan --symbols BTC/USDT ETH/USDT SOL/USDT --as-of auto
```

```
data/{exchange}/cointegration/scan_YYYY-MM-DD_{tf}.parquet
data/{exchange}/cointegration/scan_latest_{tf}.parquet
```

## Test

```bash
pytest -q                 # unit (network yok)
pytest -q -m integration  # canlı BTC-ETH smoke
```

## Tasarım ilkeleri

- Strateji eşikleri (z-score vb.) burada yok — walk-forward’dan gelecek.
- Look-ahead: yalnızca kapanmış mumlar.
- Survivorship: universe tarihli snapshot; tarama `universe_as_of` ile sınırlı.

## Sonraki fazlar

3. Sinyal · 4. Risk · 5. Walk-forward · 6. Loglama · 7. Execution
