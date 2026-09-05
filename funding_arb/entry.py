"""Step 3 — delta-neutral paper entry + unpaired-leg safety.

Live exchange execution is intentionally thin: size from depth, simulate or
submit both legs, then reconcile. If one leg fills and the other does not,
flatten the filled leg (or raise an alarm) so the book never stays directional.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class LegSide(str, Enum):
    SPOT = "spot"
    PERP = "perp"


class LegStatus(str, Enum):
    FILLED = "filled"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    REJECTED = "rejected"
    FLATTENED = "flattened"


class EntryState(str, Enum):
    BALANCED = "balanced"
    UNPAIRED = "unpaired"
    ABORTED = "aborted"
    FLATTENED = "flattened"


@dataclass
class LegFill:
    leg: LegSide
    target_usdt: float
    filled_usdt: float
    status: LegStatus
    detail: str = ""

    @property
    def fill_ratio(self) -> float:
        if self.target_usdt <= 0:
            return 0.0
        return float(self.filled_usdt) / float(self.target_usdt)

    @property
    def is_complete(self) -> bool:
        return self.status == LegStatus.FILLED and self.fill_ratio >= 0.99


@dataclass
class EntryResult:
    symbol: str
    target_usdt: float
    side_hint: str
    spot: LegFill
    perp: LegFill
    state: EntryState
    actions: list[str] = field(default_factory=list)
    alarms: list[str] = field(default_factory=list)

    @property
    def residual_usdt(self) -> float:
        """Net directional exposure left after reconcile (should be ~0)."""
        return abs(self.spot.filled_usdt - self.perp.filled_usdt)


def size_notional_from_depth(
    depth_usdt: float,
    *,
    depth_fraction: float = 0.05,
    max_notional_usdt: float = 100.0,
    min_notional_usdt: float = 10.0,
) -> float:
    """Dynamic size = depth_fraction × book depth, capped for paper/small capital.

    Returns 0 when the book is too thin for even ``min_notional_usdt``.
    """
    if depth_usdt is None or depth_usdt <= 0 or depth_fraction <= 0:
        return 0.0
    raw = float(depth_usdt) * float(depth_fraction)
    sized = min(raw, float(max_notional_usdt))
    if sized < float(min_notional_usdt):
        return 0.0
    return float(sized)


def select_entry_candidate(
    screen: Any,
    *,
    require_non_floor: bool = True,
    require_depth: bool = True,
) -> dict[str, Any] | None:
    """Pick top screen row suitable for entry (prefers real premium)."""
    if screen is None or getattr(screen, "empty", True):
        return None
    frame = screen
    if require_depth and "passes_depth" in frame.columns:
        frame = frame[frame["passes_depth"]]
    if require_non_floor and "at_interest_floor" in frame.columns:
        non_floor = frame[~frame["at_interest_floor"].astype(bool)]
        if not non_floor.empty:
            frame = non_floor
        # else: fall back to floor names only if nothing else passes
    if frame.empty:
        return None
    row = frame.iloc[0]
    return row.to_dict()


def reconcile_dual_legs(
    symbol: str,
    *,
    target_usdt: float,
    side_hint: str,
    spot: LegFill,
    perp: LegFill,
    flatten_unpaired: bool = True,
    fill_tolerance: float = 0.01,
    flatten_fn: Callable[[LegFill], LegFill] | None = None,
) -> EntryResult:
    """Reconcile spot+perp fills; flatten residual if one leg is missing.

    ``flatten_fn`` is injected so unit tests can simulate a flatten without a
    live exchange. Default flatten zeroes the filled notional and marks FLATTENED.
    """

    def _default_flatten(leg: LegFill) -> LegFill:
        logger.warning(
            "FLATTEN unpaired %s leg on %s (filled=%.4f target=%.4f)",
            leg.leg.value,
            symbol,
            leg.filled_usdt,
            leg.target_usdt,
        )
        return LegFill(
            leg=leg.leg,
            target_usdt=leg.target_usdt,
            filled_usdt=0.0,
            status=LegStatus.FLATTENED,
            detail=f"auto_flatten_after_{leg.status.value}",
        )

    flatten_fn = flatten_fn or _default_flatten
    actions: list[str] = []
    alarms: list[str] = []

    spot_ok = spot.fill_ratio >= (1.0 - fill_tolerance) and spot.status in {
        LegStatus.FILLED,
        LegStatus.PARTIAL,
    }
    perp_ok = perp.fill_ratio >= (1.0 - fill_tolerance) and perp.status in {
        LegStatus.FILLED,
        LegStatus.PARTIAL,
    }
    # Treat near-full PARTIAL as ok only if both sides match.
    both_complete = spot.is_complete and perp.is_complete
    both_near = spot_ok and perp_ok and abs(spot.filled_usdt - perp.filled_usdt) <= (
        target_usdt * fill_tolerance
    )

    if both_complete or both_near:
        return EntryResult(
            symbol=symbol,
            target_usdt=target_usdt,
            side_hint=side_hint,
            spot=spot,
            perp=perp,
            state=EntryState.BALANCED,
            actions=["dual_leg_ok"],
        )

    spot_has = spot.filled_usdt > target_usdt * fill_tolerance
    perp_has = perp.filled_usdt > target_usdt * fill_tolerance

    if spot_has and perp_has and not both_near:
        # Both partial but mismatched sizes — flatten both for safety in paper.
        alarms.append("mismatched_partial_fills")
        if flatten_unpaired:
            spot = flatten_fn(spot)
            perp = flatten_fn(perp)
            actions.extend(["flatten_spot", "flatten_perp"])
            return EntryResult(
                symbol=symbol,
                target_usdt=target_usdt,
                side_hint=side_hint,
                spot=spot,
                perp=perp,
                state=EntryState.FLATTENED,
                actions=actions,
                alarms=alarms,
            )
        return EntryResult(
            symbol=symbol,
            target_usdt=target_usdt,
            side_hint=side_hint,
            spot=spot,
            perp=perp,
            state=EntryState.UNPAIRED,
            actions=actions,
            alarms=alarms + ["unpaired_mismatch_no_flatten"],
        )

    if spot_has ^ perp_has:
        filled_leg = spot if spot_has else perp
        missing_leg = perp if spot_has else spot
        alarms.append(
            f"unpaired_leg filled={filled_leg.leg.value} missing={missing_leg.leg.value} "
            f"missing_status={missing_leg.status.value}"
        )
        if flatten_unpaired:
            if spot_has:
                spot = flatten_fn(spot)
                actions.append("flatten_spot")
            if perp_has:
                perp = flatten_fn(perp)
                actions.append("flatten_perp")
            return EntryResult(
                symbol=symbol,
                target_usdt=target_usdt,
                side_hint=side_hint,
                spot=spot,
                perp=perp,
                state=EntryState.FLATTENED,
                actions=actions,
                alarms=alarms,
            )
        return EntryResult(
            symbol=symbol,
            target_usdt=target_usdt,
            side_hint=side_hint,
            spot=spot,
            perp=perp,
            state=EntryState.UNPAIRED,
            actions=actions,
            alarms=alarms + ["unpaired_no_flatten"],
        )

    # Neither leg filled.
    actions.append("abort_no_fills")
    return EntryResult(
        symbol=symbol,
        target_usdt=target_usdt,
        side_hint=side_hint,
        spot=spot,
        perp=perp,
        state=EntryState.ABORTED,
        actions=actions,
        alarms=alarms,
    )


def paper_dual_leg_entry(
    symbol: str,
    *,
    depth_usdt: float,
    side_hint: str,
    depth_fraction: float = 0.05,
    max_notional_usdt: float = 100.0,
    min_notional_usdt: float = 10.0,
    spot_fill_ratio: float = 1.0,
    perp_fill_ratio: float = 1.0,
    spot_status: LegStatus | None = None,
    perp_status: LegStatus | None = None,
    flatten_unpaired: bool = True,
) -> EntryResult:
    """Paper entry: size from depth, apply synthetic fill ratios, reconcile.

    Inject ``spot_fill_ratio`` / ``perp_fill_ratio`` < 1 to simulate partial /
    timeout legs in unit tests.
    """
    target = size_notional_from_depth(
        depth_usdt,
        depth_fraction=depth_fraction,
        max_notional_usdt=max_notional_usdt,
        min_notional_usdt=min_notional_usdt,
    )
    if target <= 0:
        empty = LegFill(LegSide.SPOT, 0.0, 0.0, LegStatus.REJECTED, "size_zero")
        empty_p = LegFill(LegSide.PERP, 0.0, 0.0, LegStatus.REJECTED, "size_zero")
        return EntryResult(
            symbol=symbol,
            target_usdt=0.0,
            side_hint=side_hint,
            spot=empty,
            perp=empty_p,
            state=EntryState.ABORTED,
            actions=["reject_size_below_min"],
            alarms=["depth_too_thin"],
        )

    def _status(ratio: float, override: LegStatus | None) -> LegStatus:
        if override is not None:
            return override
        if ratio >= 0.99:
            return LegStatus.FILLED
        if ratio > 0:
            return LegStatus.PARTIAL
        return LegStatus.TIMEOUT

    spot = LegFill(
        leg=LegSide.SPOT,
        target_usdt=target,
        filled_usdt=target * max(0.0, min(1.0, spot_fill_ratio)),
        status=_status(spot_fill_ratio, spot_status),
    )
    perp = LegFill(
        leg=LegSide.PERP,
        target_usdt=target,
        filled_usdt=target * max(0.0, min(1.0, perp_fill_ratio)),
        status=_status(perp_fill_ratio, perp_status),
    )
    return reconcile_dual_legs(
        symbol,
        target_usdt=target,
        side_hint=side_hint,
        spot=spot,
        perp=perp,
        flatten_unpaired=flatten_unpaired,
    )
