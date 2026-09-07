"""Parquet storage for perpetual funding-rate history."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pairs_trading.data.storage import symbol_to_filename

FUNDING_COLUMNS = ["timestamp", "funding_rate", "symbol"]


class FundingStore:
    """One parquet file per swap symbol under data/{exchange}/funding/."""

    def __init__(self, root: Path, exchange_id: str) -> None:
        self.root = Path(root)
        self.exchange_id = exchange_id
        self.base_dir = self.root / exchange_id / "funding"
        # Do not mkdir here — screen/show probe multiple venues; create on write.

    def path_for(self, symbol: str) -> Path:
        return self.base_dir / f"{symbol_to_filename(symbol)}.parquet"

    def exists(self, symbol: str) -> bool:
        return self.path_for(symbol).is_file()

    def load(self, symbol: str) -> pd.DataFrame:
        path = self.path_for(symbol)
        if not path.is_file():
            return pd.DataFrame(columns=FUNDING_COLUMNS)
        return _normalize(pd.read_parquet(path))

    def latest_timestamp_ms(self, symbol: str) -> int | None:
        df = self.load(symbol)
        if df.empty:
            return None
        return int(df["timestamp"].max())

    def save(self, symbol: str, df: pd.DataFrame) -> Path:
        clean = _normalize(df)
        path = self.path_for(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        clean.to_parquet(path, index=False)
        return path

    def upsert(self, symbol: str, new_rows: pd.DataFrame) -> pd.DataFrame:
        incoming = _normalize(new_rows)
        if incoming.empty:
            return self.load(symbol)
        existing = self.load(symbol)
        if existing.empty:
            merged = incoming
        else:
            merged = pd.concat([existing, incoming], ignore_index=True)
            merged = _normalize(merged)
        self.save(symbol, merged)
        return merged

    def list_symbols(self) -> list[str]:
        if not self.base_dir.is_dir():
            return []
        return sorted(p.stem for p in self.base_dir.glob("*.parquet"))


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=FUNDING_COLUMNS)
    out = df.copy()
    for col in FUNDING_COLUMNS:
        if col not in out.columns:
            raise ValueError(f"Funding frame missing column {col}")
    out = out[FUNDING_COLUMNS]
    out["timestamp"] = out["timestamp"].astype("int64")
    out["funding_rate"] = out["funding_rate"].astype(float)
    out["symbol"] = out["symbol"].astype(str)
    out = out.dropna(subset=["timestamp", "funding_rate"])
    out = out.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return out.reset_index(drop=True)
