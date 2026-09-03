# pairs-trading

Kointegrasyon bazlı istatistiksel arbitraj (pairs trading) sistemi — kripto piyasaları.

## Faz 1 — Veri katmanı

CCXT ile Binance / Bybit / OKX’ten üst coin evrenini seçer, OHLCV geçmişini çeker ve parquet olarak saklar.

> Not: Bazı bulut IP’lerinde Binance `451` / Bybit `403` dönebilir. Bu durumda `config/default.yaml` içinde `exchange.id: okx` kullanın (`config/smoke.yaml` örneğine bakın).

### Kurulum

```bash
pip install -e ".[dev]"
```

### Komutlar

```bash
# Global flags (--config, -v) BEFORE the subcommand:
pairs-data --config config/smoke.yaml universe --print
pairs-data --config config/smoke.yaml sync --symbols BTC/USDT ETH/USDT --no-universe --full
pairs-data --config config/smoke.yaml show BTC/USDT
```

### Depolama düzeni

```
data/
  {exchange}/
    universe/{YYYY-MM-DD}.json
    {timeframe}/{SYMBOL}.parquet
```

### Tasarım ilkeleri (Faz 1)

- **Survivorship bias:** Universe günlük JSON snapshot olarak tarihlenir; backtest’te o güne ait snapshot kullanılmalı.
- **Look-ahead bias:** Yalnızca kapanmış mumlar yazılır (`closed_only`); kısmi açık mum atılır.
- **Strateji parametreleri:** Z-score / pencere eşikleri burada yok; walk-forward fazından gelecek.
- **Execution ayrımı:** Bu katman yalnızca market data; emir göndermez.

### Sonraki fazlar

2. Kointegrasyon tarama · 3. Sinyal · 4. Risk · 5. Walk-forward · 6. Loglama · 7. Execution
