"""Control cycle: exit first, then rotation/entry — never both in one tick."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd

from funding_arb.entry import EntryResult, EntryState, paper_dual_leg_entry, select_entry_candidate
from funding_arb.exit_rules import (
    ExitAction,
    ExitDecision,
    decide_exit,
    paper_close_position,
)
from funding_arb.position import PaperPosition
from funding_arb.rotation import RotationAction, RotationDecision, decide_rotation


class CycleAction(str, Enum):
    HOLD = "hold"
    EXIT = "exit"
    ROTATE = "rotate"
    ENTER = "enter"
    FLATTEN_ALARM = "flatten_alarm"


@dataclass
class CycleResult:
    action: CycleAction
    position_before: PaperPosition | None
    position_after: PaperPosition | None
    exit_decision: ExitDecision | None = None
    rotation_decision: RotationDecision | None = None
    exec_result: EntryResult | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "position_before": self.position_before.to_dict() if self.position_before else None,
            "position_after": self.position_after.to_dict() if self.position_after else None,
            "exit": self.exit_decision.to_dict() if self.exit_decision else None,
            "rotation": self.rotation_decision.to_dict() if self.rotation_decision else None,
            "exec_state": self.exec_result.state.value if self.exec_result else None,
            "exec_alarms": list(self.exec_result.alarms) if self.exec_result else [],
            "notes": list(self.notes),
        }


def _position_from_candidate(
    candidate: dict[str, Any],
    *,
    notional_usdt: float,
    exchange_id: str,
) -> PaperPosition:
    return PaperPosition(
        symbol=str(candidate["symbol"]),
        side_hint=str(candidate.get("side_hint") or "flat"),
        notional_usdt=float(notional_usdt),
        entry_funding_rate=float(candidate.get("funding_rate") or 0.0),
        entry_premium_rate=float(candidate.get("premium_rate") or 0.0),
        entry_at_floor=bool(candidate.get("at_interest_floor", False)),
        exchange_id=exchange_id,
    )


def run_control_cycle(
    screen: pd.DataFrame,
    position: PaperPosition | None,
    *,
    exchange_id: str = "okx",
    min_edge_multiple: float = 3.0,
    rotation_cost: float = 0.004,
    require_non_floor: bool = True,
    exit_on_floor: bool = True,
    exit_on_adverse_funding: bool = True,
    depth_fraction: float = 0.05,
    max_notional_usdt: float = 100.0,
    min_notional_usdt: float = 10.0,
    flatten_unpaired: bool = True,
    close_spot_fill_ratio: float = 1.0,
    close_perp_fill_ratio: float = 1.0,
    open_spot_fill_ratio: float = 1.0,
    open_perp_fill_ratio: float = 1.0,
    enable_rotation: bool = True,
    min_abs_premium: float | None = None,
) -> CycleResult:
    """One control tick.

    Order (enforced):
      1) flat → optional non-floor enter
      2) open → **exit rules first**
      3) else rotation (3× non-floor premium) if ``enable_rotation``
      4) never emit exit+rotate together

    ``min_abs_premium`` gates flat→enter only (exit rules unchanged).
    """
    notes: list[str] = []

    if position is None:
        cand = select_entry_candidate(
            screen,
            require_non_floor=require_non_floor,
            require_depth=True,
            min_abs_premium=min_abs_premium,
        )
        if cand is None:
            return CycleResult(
                action=CycleAction.HOLD,
                position_before=None,
                position_after=None,
                notes=["flat_no_candidate"],
            )
        opened = paper_dual_leg_entry(
            str(cand["symbol"]),
            depth_usdt=float(cand.get("depth_usdt") or 0.0),
            side_hint=str(cand.get("side_hint") or "flat"),
            depth_fraction=depth_fraction,
            max_notional_usdt=max_notional_usdt,
            min_notional_usdt=min_notional_usdt,
            spot_fill_ratio=open_spot_fill_ratio,
            perp_fill_ratio=open_perp_fill_ratio,
            flatten_unpaired=flatten_unpaired,
        )
        if opened.state != EntryState.BALANCED:
            return CycleResult(
                action=CycleAction.FLATTEN_ALARM,
                position_before=None,
                position_after=None,
                exec_result=opened,
                notes=["enter_not_balanced"],
            )
        pos = _position_from_candidate(
            cand, notional_usdt=opened.target_usdt, exchange_id=exchange_id
        )
        return CycleResult(
            action=CycleAction.ENTER,
            position_before=None,
            position_after=pos,
            exec_result=opened,
            notes=["entered_non_floor" if require_non_floor else "entered"],
        )

    # EXIT before ROTATION
    exit_decision = decide_exit(
        position,
        screen,
        exit_on_floor=exit_on_floor,
        exit_on_adverse_funding=exit_on_adverse_funding,
    )
    if exit_decision.action == ExitAction.EXIT:
        depth = float((exit_decision.mark or {}).get("depth_usdt") or 0.0)
        closed = paper_close_position(
            position,
            depth_usdt=depth,
            spot_fill_ratio=close_spot_fill_ratio,
            perp_fill_ratio=close_perp_fill_ratio,
            flatten_unpaired=flatten_unpaired,
            max_notional_usdt=position.notional_usdt,
        )
        notes.append(f"exit:{exit_decision.reason.value}")
        notes.append("rotation_skipped_after_exit")
        action = CycleAction.EXIT
        if closed.state == EntryState.FLATTENED:
            action = CycleAction.FLATTEN_ALARM
            notes.append("close_unpaired_flattened")
        elif closed.state != EntryState.BALANCED:
            action = CycleAction.FLATTEN_ALARM
            notes.append(f"close_state={closed.state.value}")
        return CycleResult(
            action=action,
            position_before=position,
            position_after=None,
            exit_decision=exit_decision,
            exec_result=closed,
            notes=notes,
        )

    if not enable_rotation:
        return CycleResult(
            action=CycleAction.HOLD,
            position_before=position,
            position_after=position,
            exit_decision=exit_decision,
            notes=notes + ["hold", "rotation_disabled"],
        )

    rotation = decide_rotation(
        position,
        screen,
        min_edge_multiple=min_edge_multiple,
        rotation_cost=rotation_cost,
        require_non_floor=require_non_floor,
        prefer_premium=True,
    )
    if rotation.action != RotationAction.ROTATE or rotation.candidate is None:
        return CycleResult(
            action=CycleAction.HOLD,
            position_before=position,
            position_after=position,
            exit_decision=exit_decision,
            rotation_decision=rotation,
            notes=notes + ["hold"] + list(rotation.reasons),
        )

    depth_close = 0.0
    if screen is not None and not screen.empty and "symbol" in screen.columns:
        hit = screen[screen["symbol"] == position.symbol]
        if not hit.empty:
            depth_close = float(hit.iloc[0].get("depth_usdt") or 0.0)

    closed = paper_close_position(
        position,
        depth_usdt=depth_close,
        spot_fill_ratio=close_spot_fill_ratio,
        perp_fill_ratio=close_perp_fill_ratio,
        flatten_unpaired=flatten_unpaired,
        max_notional_usdt=position.notional_usdt,
    )
    if closed.state != EntryState.BALANCED:
        return CycleResult(
            action=CycleAction.FLATTEN_ALARM,
            position_before=position,
            position_after=None if closed.state == EntryState.FLATTENED else position,
            exit_decision=exit_decision,
            rotation_decision=rotation,
            exec_result=closed,
            notes=notes + ["rotate_aborted_close_not_balanced"] + list(rotation.reasons),
        )

    cand = rotation.candidate
    opened = paper_dual_leg_entry(
        str(cand["symbol"]),
        depth_usdt=float(cand.get("depth_usdt") or 0.0),
        side_hint=str(cand.get("side_hint") or "flat"),
        depth_fraction=depth_fraction,
        max_notional_usdt=max_notional_usdt,
        min_notional_usdt=min_notional_usdt,
        spot_fill_ratio=open_spot_fill_ratio,
        perp_fill_ratio=open_perp_fill_ratio,
        flatten_unpaired=flatten_unpaired,
    )
    if opened.state != EntryState.BALANCED:
        return CycleResult(
            action=CycleAction.FLATTEN_ALARM,
            position_before=position,
            position_after=None,
            exit_decision=exit_decision,
            rotation_decision=rotation,
            exec_result=opened,
            notes=notes + ["rotate_close_ok_open_failed"] + list(rotation.reasons),
        )

    new_pos = _position_from_candidate(
        cand, notional_usdt=opened.target_usdt, exchange_id=exchange_id
    )
    return CycleResult(
        action=CycleAction.ROTATE,
        position_before=position,
        position_after=new_pos,
        exit_decision=exit_decision,
        rotation_decision=rotation,
        exec_result=opened,
        notes=notes + ["rotated"] + list(rotation.reasons),
    )
