"""Step 5 — exit rules (adverse funding / interest-floor) + paper close."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd

from funding_arb.entry import EntryResult, paper_dual_leg_entry
from funding_arb.position import PaperPosition, funding_pays_position


class ExitReason(str, Enum):
    NONE = "none"
    ADVERSE_FUNDING = "adverse_funding"
    INTEREST_FLOOR = "interest_floor"
    MISSING_MARK = "missing_mark"


class ExitAction(str, Enum):
    HOLD = "hold"
    EXIT = "exit"


@dataclass
class ExitDecision:
    action: ExitAction
    reason: ExitReason
    position: PaperPosition
    mark: dict[str, Any] | None
    funding_rate: float | None
    at_interest_floor: bool | None
    details: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "reason": self.reason.value,
            "symbol": self.position.symbol,
            "side_hint": self.position.side_hint,
            "funding_rate": self.funding_rate,
            "at_interest_floor": self.at_interest_floor,
            "details": list(self.details),
        }


def _mark_for_position(
    position: PaperPosition, screen: pd.DataFrame
) -> dict[str, Any] | None:
    if screen is None or getattr(screen, "empty", True) or "symbol" not in screen.columns:
        return None
    hit = screen[screen["symbol"] == position.symbol]
    if hit.empty:
        return None
    return hit.iloc[0].to_dict()


def decide_exit(
    position: PaperPosition,
    screen: pd.DataFrame,
    *,
    exit_on_floor: bool = True,
    exit_on_adverse_funding: bool = True,
) -> ExitDecision:
    """Exit when funding no longer pays the side, or premium collapses to floor.

    short_perp_long_spot → needs funding_rate > 0
    long_perp_short_spot → needs funding_rate < 0
    at_interest_floor → structural interest only, treat as no edge
    """
    mark = _mark_for_position(position, screen)
    if mark is None:
        return ExitDecision(
            action=ExitAction.HOLD,
            reason=ExitReason.MISSING_MARK,
            position=position,
            mark=None,
            funding_rate=None,
            at_interest_floor=None,
            details=["mark_missing_hold_until_next_cycle"],
        )

    try:
        funding_rate = float(mark.get("funding_rate"))
    except (TypeError, ValueError):
        funding_rate = None
    at_floor = bool(mark.get("at_interest_floor", False))
    details: list[str] = []

    if exit_on_adverse_funding and funding_rate is not None:
        if not funding_pays_position(position.side_hint, funding_rate):
            details.append(
                f"funding_rate={funding_rate:.8g} adverse for {position.side_hint}"
            )
            return ExitDecision(
                action=ExitAction.EXIT,
                reason=ExitReason.ADVERSE_FUNDING,
                position=position,
                mark=mark,
                funding_rate=funding_rate,
                at_interest_floor=at_floor,
                details=details,
            )

    if exit_on_floor and at_floor:
        details.append("at_interest_floor_premium_gone")
        return ExitDecision(
            action=ExitAction.EXIT,
            reason=ExitReason.INTEREST_FLOOR,
            position=position,
            mark=mark,
            funding_rate=funding_rate,
            at_interest_floor=at_floor,
            details=details,
        )

    details.append("edge_intact")
    return ExitDecision(
        action=ExitAction.HOLD,
        reason=ExitReason.NONE,
        position=position,
        mark=mark,
        funding_rate=funding_rate,
        at_interest_floor=at_floor,
        details=details,
    )


def paper_close_position(
    position: PaperPosition,
    *,
    depth_usdt: float = 0.0,
    spot_fill_ratio: float = 1.0,
    perp_fill_ratio: float = 1.0,
    flatten_unpaired: bool = True,
    depth_fraction: float = 1.0,
    max_notional_usdt: float | None = None,
    min_notional_usdt: float = 0.0,
) -> EntryResult:
    """Paper close = reverse dual-leg with the same unpaired-leg safety as entry."""
    max_n = float(
        max_notional_usdt if max_notional_usdt is not None else position.notional_usdt
    )
    synthetic_depth = (
        position.notional_usdt / depth_fraction
        if depth_fraction > 0
        else position.notional_usdt
    )
    if depth_usdt > 0:
        synthetic_depth = max(synthetic_depth, float(depth_usdt))
    return paper_dual_leg_entry(
        position.symbol,
        depth_usdt=synthetic_depth,
        side_hint=f"close:{position.side_hint}",
        depth_fraction=depth_fraction,
        max_notional_usdt=max_n,
        min_notional_usdt=min_notional_usdt,
        spot_fill_ratio=spot_fill_ratio,
        perp_fill_ratio=perp_fill_ratio,
        flatten_unpaired=flatten_unpaired,
    )
