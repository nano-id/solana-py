"""OHLCV parquet storage with incremental merge (no look-ahead writes)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def symbol_to_filename(symbol: str) -> str:
    """BTC/USDT:USDT -> BTC_USDT_USDT.parquet-safe stem."""
    return symbol.replace("/", "_").replace(":", "_").replace("-", "_")


class ParquetStore:
    """Store OHLCV bars as one parquet file per symbol."""

    def __init__(self, root: Path, exchange_id: str, timeframe: str) -> None:
        self.root = Path(root)
        self.exchange_id = exchange_id
        self.timeframe = timeframe
        self.base_dir = self.root / exchange_id / timeframe
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, symbol: str) -> Path:
        return self.base_dir / f"{symbol_to_filename(symbol)}.parquet"

    def exists(self, symbol: str) -> bool:
        return self.path_for(symbol).is_file()

    def load(self, symbol: str) -> pd.DataFrame:
        path = self.path_for(symbol)
        if not path.is_file():
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        df = pd.read_parquet(path)
        return _normalize_ohlcv(df)

    def latest_timestamp_ms(self, symbol: str) -> int | None:
        df = self.load(symbol)
        if df.empty:
            return None
        return int(df["timestamp"].max())

    def save(self, symbol: str, df: pd.DataFrame) -> Path:
        """Replace file with normalized, sorted, de-duplicated OHLCV."""
        clean = _normalize_ohlcv(df)
        path = self.path_for(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        clean.to_parquet(path, index=False)
        return path

    def upsert(self, symbol: str, new_bars: pd.DataFrame) -> pd.DataFrame:
        """Merge new bars into existing series; keep last row per timestamp."""
        incoming = _normalize_ohlcv(new_bars)
        if incoming.empty:
            return self.load(symbol)
        existing = self.load(symbol)
        if existing.empty:
            merged = incoming
        else:
            merged = pd.concat([existing, incoming], ignore_index=True)
            merged = _normalize_ohlcv(merged)
        self.save(symbol, merged)
        return merged

    def list_symbols(self) -> list[str]:
        return sorted(p.stem for p in self.base_dir.glob("*.parquet"))


def ohlcv_rows_to_frame(rows: list[list[float | int]]) -> pd.DataFrame:
    """Convert CCXT OHLCV rows [[ts, o, h, l, c, v], ...] to DataFrame."""
    if not rows:
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    df = pd.DataFrame(rows, columns=OHLCV_COLUMNS)
    return _normalize_ohlcv(df)


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    out = df.copy()
    missing = [c for c in OHLCV_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"OHLCV missing columns: {missing}")
    out = out[OHLCV_COLUMNS]
    out["timestamp"] = out["timestamp"].astype("int64")
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["timestamp", "open", "high", "low", "close"])
    out = out.drop_duplicates(subset=["timestamp"], keep="last")
    out = out.sort_values("timestamp").reset_index(drop=True)
    # Attach UTC datetime for readability without using it as index (avoids
    # accidental look-ahead via index alignment bugs in later modules).
    out["datetime_utc"] = pd.to_datetime(out["timestamp"], unit="ms", utc=True)
    return out
