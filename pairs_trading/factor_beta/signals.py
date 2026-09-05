"""Step 4 — z-score signal skeleton (per Rule-5e cluster), mechanical only.

Constraints baked in from day one:
- entry/exit z are SKELETON defaults for OU mechanical tests only;
  live calibration waits for walk-forward (Step 6). Do not treat as proven.
- Rule 5b edge/cost gate is a required placeholder in the entry API signature
  (not deferred / not forgettable), even though Step 4 does not yet compute
  real expected edge.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from pairs_trading.factor_beta.config import FactorBetaConfig, load_factor_beta_config
from pairs_trading.factor_beta.factor_model import fit_rolling_factor
from pairs_trading.signal.engine import Side, SignalConfig, generate_signals
from pairs_trading.signal.ou import simulate_ou_spread


@dataclass(frozen=True)
class EntryGatePlaceholder:
    """Rule 5b placeholder — signature presence is mandatory.

    Step 4 does not compute real expected edge yet; callers must still pass
    this object so the cost gate cannot be silently omitted later.
    """

    min_edge_to_cost_multiple: float
    round_trip_bps: float
    expected_edge_bps: float | None = None  # None ⇒ gate not evaluable yet

    def allows_entry(self) -> tuple[bool, str]:
        if self.expected_edge_bps is None:
            return True, "edge_not_computed_yet_placeholder_pass"
        need = self.min_edge_to_cost_multiple * self.round_trip_bps
        if self.expected_edge_bps >= need:
            return True, "edge_covers_cost_gate"
        return False, f"edge {self.expected_edge_bps:.2f}bps < {need:.2f}bps required"


def load_signal_lines(path: Path | None = None) -> dict:
    path = path or Path("data/okx/factor_beta/step5e_signal_lines.json")
    return json.loads(path.read_text(encoding="utf-8"))


def skeleton_signal_config(config: FactorBetaConfig) -> SignalConfig:
    """Pick midpoint of configured grids — SKELETON ONLY, not calibrated."""
    z_entry = float(sorted(config.signal.z_entry_grid)[len(config.signal.z_entry_grid) // 2])
    z_exit = float(sorted(config.signal.z_exit_grid)[len(config.signal.z_exit_grid) // 2])
    z_win = int(sorted(config.signal.zscore_window_grid)[len(config.signal.zscore_window_grid) // 2])
    # stop: loose placeholder above max entry (not a tuned risk limit)
    stop = max(config.signal.z_entry_grid) + 1.0
    return SignalConfig(entry_z=z_entry, exit_z=z_exit, stop_z=stop, z_window=z_win)


def maybe_enter(
    *,
    z: float,
    side_flat: bool,
    signal_cfg: SignalConfig,
    gate: EntryGatePlaceholder,
) -> tuple[bool, str, Literal["enter_long", "enter_short", "blocked", "none"]]:
    """Entry helper with Rule-5b gate in the signature (placeholder-ready)."""
    if not side_flat:
        return False, "not_flat", "none"
    if abs(z) < signal_cfg.entry_z:
        return False, "below_entry_z", "none"
    ok, reason = gate.allows_entry()
    if not ok:
        return False, reason, "blocked"
    if z <= -signal_cfg.entry_z:
        return True, reason, "enter_long"
    return True, reason, "enter_short"


def run_cluster_signal_skeleton(
    config: FactorBetaConfig | None = None,
    *,
    lookback_days: int | None = None,
) -> dict:
    """Build one z-signal line per Rule-5e cluster (representative residual)."""
    config = config or load_factor_beta_config()
    lookback = int(lookback_days or config.factor.beta_lookback_days_default)
    lines_doc = load_signal_lines()
    signal_cfg = skeleton_signal_config(config)
    gate = EntryGatePlaceholder(
        min_edge_to_cost_multiple=float(config.entry_gate.min_edge_to_cost_multiple),
        round_trip_bps=float(config.costs.round_trip_bps),
        expected_edge_bps=None,  # Step 4: not computed
    )

    result = fit_rolling_factor(config, lookback_days=lookback)
    per_line: list[dict] = []

    for line in lines_doc["signal_lines"]:
        sym = line["symbol"]
        resid = result.residual[sym].dropna()
        frame, events = generate_signals(resid, signal_cfg)
        # Count entries that would pass / block under placeholder gate
        n_enter = sum(1 for e in events if e.reason in ("enter_long", "enter_short"))
        n_exit_mean = sum(1 for e in events if e.reason == "exit_mean")
        n_exit_stop = sum(1 for e in events if e.reason == "exit_stop")
        # Demonstrate gate hook on last bar if flat+extreme z
        z_last = float(frame["z"].dropna().iloc[-1]) if frame["z"].notna().any() else float("nan")
        side_last = frame["side"].iloc[-1] if len(frame) else Side.FLAT.value
        allowed, gate_reason, action = maybe_enter(
            z=z_last if np.isfinite(z_last) else 0.0,
            side_flat=(side_last == Side.FLAT.value),
            signal_cfg=signal_cfg,
            gate=gate,
        )
        per_line.append(
            {
                "signal_id": line["signal_id"],
                "cluster_id": line["cluster_id"],
                "symbol": sym,
                "members_covered": line["members_covered"],
                "n_residual_bars": int(len(resid)),
                "n_enter_events": n_enter,
                "n_exit_mean": n_exit_mean,
                "n_exit_stop": n_exit_stop,
                "z_last": z_last if np.isfinite(z_last) else None,
                "side_last": side_last,
                "rule5b_gate_placeholder": {
                    "min_edge_to_cost_multiple": gate.min_edge_to_cost_multiple,
                    "round_trip_bps": gate.round_trip_bps,
                    "expected_edge_bps": gate.expected_edge_bps,
                    "last_bar_gate_allows": allowed,
                    "last_bar_gate_reason": gate_reason,
                    "last_bar_action": action,
                },
            }
        )

    return {
        "step": 4,
        "step_name": "cluster_zscore_signal_skeleton",
        "status": "complete",
        "interpretation_allowed": False,
        "lookback_days": lookback,
        "n_signal_lines": len(per_line),
        "skeleton_thresholds_NOT_calibrated": {
            "entry_z": signal_cfg.entry_z,
            "exit_z": signal_cfg.exit_z,
            "stop_z": signal_cfg.stop_z,
            "z_window": signal_cfg.z_window,
            "source": "midpoint of config grids — walk-forward (Step 6) owns live values",
            "z_entry_grid": list(config.signal.z_entry_grid),
            "z_exit_grid": list(config.signal.z_exit_grid),
            "zscore_window_grid": list(config.signal.zscore_window_grid),
        },
        "rule5b_placeholder_present": True,
        "per_line": per_line,
        "rules_still_locked": True,
        "note": (
            "Mechanical skeleton only. Thresholds uncalibrated. "
            "Rule 5b gate is in the entry signature but expected_edge_bps is None. "
            "No PnL / no edge claim."
        ),
        "next_step": 6,
        "next_step_name": "walk_forward_threshold_calibration",
        "optional_parallel": "step7_synthetic_ou_mechanical_test",
    }


def run_synthetic_ou_mechanical_test(
    config: FactorBetaConfig | None = None,
    *,
    n: int = 3000,
    seed: int = 7,
) -> dict:
    """Verify enter/exit wiring on a known mean-reverting OU path."""
    config = config or load_factor_beta_config()
    signal_cfg = skeleton_signal_config(config)
    gate = EntryGatePlaceholder(
        min_edge_to_cost_multiple=float(config.entry_gate.min_edge_to_cost_multiple),
        round_trip_bps=float(config.costs.round_trip_bps),
        expected_edge_bps=None,
    )
    spread = simulate_ou_spread(n=n, theta=0.12, mu=0.0, sigma=0.04, seed=seed)
    frame, events = generate_signals(spread, signal_cfg)
    n_enter = sum(1 for e in events if e.reason in ("enter_long", "enter_short"))
    n_exit = sum(1 for e in events if e.reason in ("exit_mean", "exit_stop"))
    # Gate must be callable without throwing
    _ok, _reason, _ = maybe_enter(
        z=-signal_cfg.entry_z - 0.1,
        side_flat=True,
        signal_cfg=signal_cfg,
        gate=gate,
    )
    return {
        "test": "synthetic_ou_mechanical",
        "passed": n_enter > 0 and n_exit > 0 and _ok is True,
        "n_bars": n,
        "n_enter": n_enter,
        "n_exit": n_exit,
        "entry_z_skeleton": signal_cfg.entry_z,
        "exit_z_skeleton": signal_cfg.exit_z,
        "rule5b_placeholder_invoked": True,
        "note": "Mechanical wiring check only — not a strategy performance result.",
    }


def save_step4_report(report: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "step4_cluster_zscore_skeleton.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
