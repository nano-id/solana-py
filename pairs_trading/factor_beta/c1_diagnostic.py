"""C1 cluster diagnostic — commonality + composite residual vs ADA rep.

Not a trade step. Decides whether Step-6 C1 uses ADA alone or equal-weight
composite of {ADA, APT, FIL, LTC, SUI}.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pairs_trading.factor_beta.config import FactorBetaConfig, load_factor_beta_config
from pairs_trading.factor_beta.factor_model import fit_rolling_factor
from pairs_trading.factor_beta.stationarity import _adf_one

# Research tags only — descriptive, not a claim of economic cointegration.
C1_MEMBER_PROFILES: dict[str, dict] = {
    "ADA/USDT": {
        "name": "Cardano",
        "sector": "L1_smart_contract",
        "family": "UTXO_L1",
        "narrative": "legacy smart-contract L1 (non-EVM)",
    },
    "APT/USDT": {
        "name": "Aptos",
        "sector": "L1_smart_contract",
        "family": "Move_L1",
        "narrative": "Move-language L1 (newer)",
    },
    "SUI/USDT": {
        "name": "Sui",
        "sector": "L1_smart_contract",
        "family": "Move_L1",
        "narrative": "Move-language L1 (newer)",
    },
    "FIL/USDT": {
        "name": "Filecoin",
        "sector": "decentralized_storage",
        "family": "storage_infra",
        "narrative": "storage / infra — not a trading L1 peer",
    },
    "LTC/USDT": {
        "name": "Litecoin",
        "sector": "payments_legacy_L1",
        "family": "BTC_fork_payments",
        "narrative": "legacy payments L1 / BTC peer — different era",
    },
}


def equal_weight_composite(residuals: pd.DataFrame, members: list[str]) -> pd.Series:
    frame = residuals[members].dropna(how="any")
    return frame.mean(axis=1).rename("C1_composite")


def run_c1_diagnostic(
    config: FactorBetaConfig | None = None,
    *,
    lookback_days: int | None = None,
    members: list[str] | None = None,
) -> dict:
    """Sector commonality + ADF(ADA) vs ADF(equal-weight composite)."""
    config = config or load_factor_beta_config()
    lookback = int(lookback_days or config.factor.beta_lookback_days_default)
    members = members or [
        "ADA/USDT",
        "APT/USDT",
        "FIL/USDT",
        "LTC/USDT",
        "SUI/USDT",
    ]
    result = fit_rolling_factor(config, lookback_days=lookback)
    missing = [m for m in members if m not in result.residual.columns]
    if missing:
        raise KeyError(f"C1 members missing from residuals: {missing}")

    resid = result.residual[members].dropna(how="any")
    ada = resid["ADA/USDT"]
    composite = equal_weight_composite(result.residual, members)

    adf_ada = _adf_one(ada)
    adf_comp = _adf_one(composite)
    pmax = float(config.stationarity.adf_pvalue_max)
    ada_pass = bool(
        adf_ada["status"] == "ok"
        and adf_ada["adf_pvalue"] is not None
        and adf_ada["adf_pvalue"] <= pmax
    )
    comp_pass = bool(
        adf_comp["status"] == "ok"
        and adf_comp["adf_pvalue"] is not None
        and adf_comp["adf_pvalue"] <= pmax
    )

    # Noise proxy: residual std + ADF p (lower p = stronger stationarity reject of unit root)
    std_ada = float(ada.std(ddof=0))
    std_comp = float(composite.std(ddof=0))
    p_ada = float(adf_ada["adf_pvalue"]) if adf_ada["adf_pvalue"] is not None else 9.0
    p_comp = float(adf_comp["adf_pvalue"]) if adf_comp["adf_pvalue"] is not None else 9.0

    sectors = sorted({C1_MEMBER_PROFILES[m]["sector"] for m in members})
    families = sorted({C1_MEMBER_PROFILES[m]["family"] for m in members})
    # Overlap: how many share the modal sector
    sector_counts: dict[str, int] = {}
    for m in members:
        s = C1_MEMBER_PROFILES[m]["sector"]
        sector_counts[s] = sector_counts.get(s, 0) + 1
    modal_sector, modal_n = max(sector_counts.items(), key=lambda kv: kv[1])

    if modal_n == len(members):
        commonality = "homogeneous_sector"
    elif modal_n >= 3:
        commonality = "partial_overlap_not_homogeneous"
    else:
        commonality = "heterogeneous_no_clear_sector"

    # Recommendation: prefer composite if it passes ADF and is at least as significant
    # (lower or equal p) OR meaningfully less volatile while still passing.
    prefer_composite = False
    reason_parts: list[str] = []
    if comp_pass and (not ada_pass):
        prefer_composite = True
        reason_parts.append("composite_passes_ADF_ADA_fails")
    elif comp_pass and ada_pass:
        if p_comp <= p_ada:
            prefer_composite = True
            reason_parts.append("composite_adf_pvalue_leq_ADA")
        if std_comp < std_ada * 0.95:
            prefer_composite = True
            reason_parts.append("composite_std_lower")
        if not prefer_composite:
            # still lean composite for averaging if both pass and p within 2x
            if p_comp <= max(p_ada * 2.0, p_ada + 0.01):
                prefer_composite = True
                reason_parts.append("both_pass_composite_comparable_noise_avg_preferred")
    elif ada_pass and not comp_pass:
        prefer_composite = False
        reason_parts.append("keep_ADA_composite_fails_ADF")
    else:
        prefer_composite = False
        reason_parts.append("neither_clear_keep_ADA_rep")

    if not reason_parts:
        reason_parts.append("default_ADA_rep")

    profiles = {m: C1_MEMBER_PROFILES[m] for m in members}
    pairwise = {
        a: {b: round(float(resid[a].corr(resid[b])), 4) for b in members} for a in members
    }

    return {
        "step": "c1_diagnostic",
        "step_name": "c1_commonality_and_composite_residual",
        "status": "complete",
        "interpretation_allowed": False,
        "lookback_days": lookback,
        "members": members,
        "profiles": profiles,
        "sector_counts": sector_counts,
        "sectors_present": sectors,
        "families_present": families,
        "commonality_label": commonality,
        "commonality_note": (
            f"{modal_n}/{len(members)} share modal sector '{modal_sector}'. "
            "APT+SUI share Move-L1 family; ADA is separate L1 family; "
            "FIL=storage; LTC=legacy payments. Not a single clean sector/narrative."
        ),
        "n_residual_bars": int(len(resid)),
        "pairwise_residual_corr": pairwise,
        "ada_representative": {
            "symbol": "ADA/USDT",
            "resid_std": std_ada,
            "adf": adf_ada,
            "passes_adf": ada_pass,
        },
        "equal_weight_composite": {
            "members": members,
            "resid_std": std_comp,
            "std_ratio_vs_ada": round(std_comp / std_ada, 4) if std_ada > 0 else None,
            "adf": adf_comp,
            "passes_adf": comp_pass,
        },
        "comparison": {
            "adf_pvalue_ada": p_ada,
            "adf_pvalue_composite": p_comp,
            "composite_more_significant": p_comp < p_ada,
            "composite_less_volatile": std_comp < std_ada,
        },
        "recommendation": {
            "c1_residual_source": "equal_weight_composite" if prefer_composite else "representative_ADA",
            "prefer_composite": prefer_composite,
            "reasons": reason_parts,
        },
        "rules_still_locked": True,
        "note": (
            "Diagnostic only. Economic commonality is partial/heterogeneous — "
            "treat C1 as a statistical residual cluster, not a proven sector factor. "
            "No PnL / no edge claim."
        ),
        "next_step": 6,
        "next_step_name": "per_line_walk_forward_calibration",
    }


def save_c1_diagnostic(report: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "step_c1_diagnostic.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
