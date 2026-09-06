"""Step 4 — rotate only into non-floor names that clear a premium multiple."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd

from funding_arb.position import PaperPosition, edge_score


class RotationAction(str, Enum):
    HOLD = "hold"
    ROTATE = "rotate"
    SKIP = "skip"


def _best_non_floor_candidate(
    screen: pd.DataFrame,
    *,
    require_non_floor: bool,
    prefer_premium: bool,
) -> dict[str, Any] | None:
    """Best depth-passing name by |premium|; never prefer interest-floor when avoidable."""
    if screen is None or getattr(screen, "empty", True):
        return None
    frame = screen.copy()
    if "passes_depth" in frame.columns:
        frame = frame[frame["passes_depth"].astype(bool)]
    if frame.empty:
        return None
    if require_non_floor and "at_interest_floor" in frame.columns:
        non_floor = frame[~frame["at_interest_floor"].astype(bool)]
        if not non_floor.empty:
            frame = non_floor
        else:
            return None
    if frame.empty:
        return None
    scored = [
        (edge_score(row.to_dict(), prefer_premium=prefer_premium), i)
        for i, row in frame.iterrows()
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    best_i = scored[0][1]
    return frame.loc[best_i].to_dict()


@dataclass
class RotationDecision:
    action: RotationAction
    current: PaperPosition | None
    candidate: dict[str, Any] | None
    current_edge: float
    candidate_edge: float
    ratio: float
    threshold: float
    rotation_cost: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "current_symbol": self.current.symbol if self.current else None,
            "candidate_symbol": (self.candidate or {}).get("symbol"),
            "current_edge": self.current_edge,
            "candidate_edge": self.candidate_edge,
            "ratio": self.ratio,
            "threshold": self.threshold,
            "rotation_cost": self.rotation_cost,
            "reasons": list(self.reasons),
        }


def decide_rotation(
    position: PaperPosition,
    screen: pd.DataFrame,
    *,
    min_edge_multiple: float = 3.0,
    rotation_cost: float = 0.004,
    require_non_floor: bool = True,
    prefer_premium: bool = True,
) -> RotationDecision:
    """Rotate iff a non-floor candidate's |premium| is ≥ multiple × current.

    Uses ``premium_rate`` (funding − interest), not raw ann_funding, so
    interest-floor names do not win on structural 0.0001.
    ``rotation_cost`` is logged for gross-vs-net honesty; the gate is the multiple.
    """
    reasons: list[str] = []
    current_row = None
    if screen is not None and not getattr(screen, "empty", True) and "symbol" in screen.columns:
        hit = screen[screen["symbol"] == position.symbol]
        if not hit.empty:
            current_row = hit.iloc[0].to_dict()

    if current_row is not None:
        current_edge = edge_score(current_row, prefer_premium=prefer_premium)
        if bool(current_row.get("at_interest_floor", False)):
            reasons.append("current_at_interest_floor")
    else:
        current_edge = abs(float(position.entry_premium_rate))
        if prefer_premium and current_edge == 0.0:
            current_edge = abs(float(position.entry_funding_rate))
        reasons.append("current_row_missing_use_entry_edge")

    candidate = _best_non_floor_candidate(
        screen,
        require_non_floor=require_non_floor,
        prefer_premium=prefer_premium,
    )
    if candidate is None:
        return RotationDecision(
            action=RotationAction.SKIP,
            current=position,
            candidate=None,
            current_edge=current_edge,
            candidate_edge=0.0,
            ratio=0.0,
            threshold=min_edge_multiple,
            rotation_cost=rotation_cost,
            reasons=reasons + ["no_non_floor_depth_candidate"],
        )

    cand_edge = edge_score(candidate, prefer_premium=prefer_premium)
    if str(candidate.get("symbol")) == position.symbol:
        return RotationDecision(
            action=RotationAction.HOLD,
            current=position,
            candidate=candidate,
            current_edge=current_edge,
            candidate_edge=cand_edge,
            ratio=1.0,
            threshold=min_edge_multiple,
            rotation_cost=rotation_cost,
            reasons=reasons + ["candidate_is_current"],
        )

    if require_non_floor and bool(candidate.get("at_interest_floor", False)):
        return RotationDecision(
            action=RotationAction.HOLD,
            current=position,
            candidate=candidate,
            current_edge=current_edge,
            candidate_edge=cand_edge,
            ratio=0.0,
            threshold=min_edge_multiple,
            rotation_cost=rotation_cost,
            reasons=reasons + ["candidate_at_interest_floor_blocked"],
        )

    ratio = (cand_edge / current_edge) if current_edge > 0 else float("inf")
    if current_edge <= 0 and cand_edge > 0:
        return RotationDecision(
            action=RotationAction.ROTATE,
            current=position,
            candidate=candidate,
            current_edge=current_edge,
            candidate_edge=cand_edge,
            ratio=float("inf"),
            threshold=min_edge_multiple,
            rotation_cost=rotation_cost,
            reasons=reasons + ["current_edge_zero_candidate_positive"],
        )

    if ratio >= float(min_edge_multiple):
        return RotationDecision(
            action=RotationAction.ROTATE,
            current=position,
            candidate=candidate,
            current_edge=current_edge,
            candidate_edge=cand_edge,
            ratio=float(ratio),
            threshold=min_edge_multiple,
            rotation_cost=rotation_cost,
            reasons=reasons + [f"ratio={ratio:.4g}>={min_edge_multiple}"],
        )

    return RotationDecision(
        action=RotationAction.HOLD,
        current=position,
        candidate=candidate,
        current_edge=current_edge,
        candidate_edge=cand_edge,
        ratio=float(ratio),
        threshold=min_edge_multiple,
        rotation_cost=rotation_cost,
        reasons=reasons + [f"ratio={ratio:.4g}<{min_edge_multiple}"],
    )
