"""Paper position state shared by rotation / exit / control cycles."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class PaperPosition:
    """Delta-neutral paper position held between control cycles."""

    symbol: str
    side_hint: str  # short_perp_long_spot | long_perp_short_spot
    notional_usdt: float
    entry_funding_rate: float
    entry_premium_rate: float
    entry_at_floor: bool
    opened_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    exchange_id: str = "okx"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PaperPosition:
        return cls(
            symbol=str(raw["symbol"]),
            side_hint=str(raw["side_hint"]),
            notional_usdt=float(raw["notional_usdt"]),
            entry_funding_rate=float(raw["entry_funding_rate"]),
            entry_premium_rate=float(raw.get("entry_premium_rate") or 0.0),
            entry_at_floor=bool(raw.get("entry_at_floor", False)),
            opened_at=str(
                raw.get("opened_at") or datetime.now(timezone.utc).isoformat()
            ),
            exchange_id=str(raw.get("exchange_id") or "okx"),
        )


def edge_score(row: dict[str, Any] | None, *, prefer_premium: bool = True) -> float:
    """Absolute edge for rotation math (premium over interest preferred)."""
    if not row:
        return 0.0
    if prefer_premium and row.get("premium_rate") is not None:
        try:
            return abs(float(row["premium_rate"]))
        except (TypeError, ValueError):
            pass
    for key in ("ann_funding", "funding_rate"):
        if row.get(key) is not None:
            try:
                return abs(float(row[key]))
            except (TypeError, ValueError):
                continue
    return 0.0


def funding_pays_position(side_hint: str, funding_rate: float) -> bool:
    """True when funding still pays the held delta-neutral side."""
    if side_hint == "short_perp_long_spot":
        return funding_rate > 0
    if side_hint == "long_perp_short_spot":
        return funding_rate < 0
    return False
