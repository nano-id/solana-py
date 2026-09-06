"""Robustness stress battery for the calibrated 1h MM parameter set.

Stresses map to previously discussed risks: fill model, adverse selection,
fees, size/liquidity, seed, universe, and calendar regime. Evaluated on the
post-calibration blind window (default: 2026-01 → 2026-08).

Not a live-edge claim. interpretation_allowed=False.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from market_making.config import MarketMakingConfig
from market_making.portfolio import run_portfolio_month


@dataclass
class StressScenario:
    name: str
    risk: str
    description: str
    mutate: Callable[[MarketMakingConfig], MarketMakingConfig]
    # Optional per-run overrides
    seed: int | None = None
    symbols: list[str] | None = None
    months: list[tuple[int, int]] | None = None


@dataclass
class StressCaseResult:
    name: str
    risk: str
    description: str
    start_equity: float
    final_equity: float
    total_return_pct: float
    n_pos_months: int
    n_neg_months: int
    n_months: int
    total_fees: float
    worst_month: dict[str, Any]
    best_month: dict[str, Any]
    monthly: list[dict[str, Any]]
    survived: bool  # final equity > start
    broken: bool  # final equity <= start OR any deep month loss threshold


@dataclass
class StressBatteryResult:
    baseline_params: dict[str, Any]
    window: list[str]
    timeframe: str
    start_equity: float
    cases: list[StressCaseResult]
    n_survived: int
    n_broken: int
    broken_names: list[str]
    interpretation_allowed: bool
    method: str
    notes: str = ""


def _with_sim(cfg: MarketMakingConfig, **kwargs: float) -> MarketMakingConfig:
    sim = cfg.sim.model_copy(update=kwargs)
    return cfg.model_copy(update={"sim": sim})


def _compound_months(
    cfg: MarketMakingConfig,
    months: list[tuple[int, int]],
    *,
    timeframe: str,
    seed: int,
    initial_equity: float,
    symbols: list[str] | None = None,
) -> dict[str, Any]:
    equity = float(initial_equity)
    monthly: list[dict[str, Any]] = []
    for year, month in months:
        pres, _ = run_portfolio_month(
            cfg,
            year=year,
            month=month,
            timeframe=timeframe,
            seed=seed,
            initial_equity=equity,
            symbols=symbols,
        )
        start = float(pres.start_equity)
        end = float(pres.final_equity)
        ret = float(pres.total_return_pct)
        monthly.append(
            {
                "ym": f"{year:04d}-{month:02d}",
                "start": start,
                "end": end,
                "ret_pct": ret,
                "fees": float(pres.fees_paid),
                "n_entries": int(pres.n_entries),
            }
        )
        equity = end

    start0 = float(monthly[0]["start"]) if monthly else float(initial_equity)
    end1 = float(monthly[-1]["end"]) if monthly else float(initial_equity)
    worst = min(monthly, key=lambda r: r["ret_pct"]) if monthly else {}
    best = max(monthly, key=lambda r: r["ret_pct"]) if monthly else {}
    return {
        "start_equity": start0,
        "final_equity": end1,
        "total_return_pct": 100.0 * (end1 - start0) / start0 if start0 else 0.0,
        "n_pos_months": int(sum(1 for r in monthly if r["ret_pct"] > 0)),
        "n_neg_months": int(sum(1 for r in monthly if r["ret_pct"] <= 0)),
        "n_months": len(monthly),
        "total_fees": float(sum(r["fees"] for r in monthly)),
        "worst_month": {"ym": worst.get("ym"), "ret_pct": worst.get("ret_pct")},
        "best_month": {"ym": best.get("ym"), "ret_pct": best.get("ret_pct")},
        "monthly": monthly,
    }


def default_blind_months() -> list[tuple[int, int]]:
    """Post-calibration blind full months available in data."""
    return [(2026, m) for m in range(1, 9)]  # Jan–Aug 2026


def build_stress_scenarios(cfg: MarketMakingConfig) -> list[StressScenario]:
    """Map discussed risks → concrete config mutations."""
    base_syms = list(cfg.symbols)
    # Drop alts that often drive paper edge
    no_meme = [s for s in base_syms if s.split("/")[0] not in {"DOGE", "SUI"}]
    majors_only = [s for s in base_syms if s.split("/")[0] in {"BTC", "ETH", "BNB", "SOL"}]

    scenarios: list[StressScenario] = [
        StressScenario(
            name="baseline",
            risk="control",
            description="Calibrated 1h params, fill_prob=0.25, adverse_frac=0.55, fee=0.07%",
            mutate=lambda c: c,
        ),
        StressScenario(
            name="fill_prob_0.15",
            risk="fill_model",
            description="Harder fills (15% touch prob)",
            mutate=lambda c: _with_sim(c, fill_prob=0.15),
        ),
        StressScenario(
            name="fill_prob_0.10",
            risk="fill_model",
            description="Very hard fills (10% touch prob)",
            mutate=lambda c: _with_sim(c, fill_prob=0.10),
        ),
        StressScenario(
            name="adverse_0.70",
            risk="adverse_selection",
            description="Worse adverse fill price (70% of half-spread)",
            mutate=lambda c: _with_sim(c, adverse_frac=0.70),
        ),
        StressScenario(
            name="adverse_0.85",
            risk="adverse_selection",
            description="Severe adverse fill price (85% of half-spread)",
            mutate=lambda c: _with_sim(c, adverse_frac=0.85),
        ),
        StressScenario(
            name="fill_adverse_harsh",
            risk="fill_model+adverse",
            description="Combined: fill_prob=0.10 and adverse_frac=0.85",
            mutate=lambda c: _with_sim(c, fill_prob=0.10, adverse_frac=0.85),
        ),
        StressScenario(
            name="fee_0.10pct",
            risk="fees",
            description="Higher taker-like fee 0.10%/side",
            mutate=lambda c: c.model_copy(update={"commission_pct": 0.10}),
        ),
        StressScenario(
            name="fee_0.15pct",
            risk="fees",
            description="Harsh fee 0.15%/side",
            mutate=lambda c: c.model_copy(update={"commission_pct": 0.15}),
        ),
        StressScenario(
            name="size_frac_0.25",
            risk="size_liquidity",
            description="Larger clips (25% of max inventory) — more inventory risk",
            mutate=lambda c: c.model_copy(update={"size_frac_of_max": 0.25}),
        ),
        StressScenario(
            name="size_frac_0.05",
            risk="size_liquidity",
            description="Smaller clips (5% of max) — fee drag / slower inventory",
            mutate=lambda c: c.model_copy(update={"size_frac_of_max": 0.05}),
        ),
        StressScenario(
            name="halt_0.50",
            risk="inventory",
            description="Earlier inventory halt (50%) — less quoting when loaded",
            mutate=lambda c: c.model_copy(update={"inventory_halt_frac": 0.50}),
        ),
        StressScenario(
            name="seed_7",
            risk="seed_sensitivity",
            description="Alternate RNG seed for fills",
            mutate=lambda c: c,
            seed=7,
        ),
        StressScenario(
            name="seed_99",
            risk="seed_sensitivity",
            description="Second alternate RNG seed",
            mutate=lambda c: c,
            seed=99,
        ),
        StressScenario(
            name="universe_no_doge_sui",
            risk="universe",
            description="Drop DOGE+SUI from universe",
            mutate=lambda c: c.model_copy(update={"symbols": no_meme}),
            symbols=no_meme,
        ),
        StressScenario(
            name="universe_majors4",
            risk="universe",
            description="Only BTC ETH BNB SOL",
            mutate=lambda c: c.model_copy(update={"symbols": majors_only}),
            symbols=majors_only,
        ),
        StressScenario(
            name="drop_best_month_feb",
            risk="regime",
            description="Blind window without best month (2026-02)",
            mutate=lambda c: c,
            months=[(2026, m) for m in (1, 3, 4, 5, 6, 7, 8)],
        ),
        StressScenario(
            name="h1_only",
            risk="regime",
            description="Only 2026 H1 (Jan–Jun)",
            mutate=lambda c: c,
            months=[(2026, m) for m in range(1, 7)],
        ),
        StressScenario(
            name="h2_only",
            risk="regime",
            description="Only 2026 H2 available (Jul–Aug)",
            mutate=lambda c: c,
            months=[(2026, 7), (2026, 8)],
        ),
        StressScenario(
            name="param_gamma_down",
            risk="param_perturbation",
            description="gamma 0.18→0.12 (less risk aversion / different AS)",
            mutate=lambda c: c.model_copy(update={"gamma": 0.12}),
        ),
        StressScenario(
            name="param_tau_16",
            risk="param_perturbation",
            description="tau_bars 48→16 (shorter horizon)",
            mutate=lambda c: c.model_copy(update={"tau_bars": 16.0}),
        ),
        StressScenario(
            name="kitchen_sink",
            risk="combined",
            description="fill=0.12, adverse=0.75, fee=0.10%, no DOGE/SUI",
            mutate=lambda c: _with_sim(
                c.model_copy(update={"commission_pct": 0.10, "symbols": no_meme}),
                fill_prob=0.12,
                adverse_frac=0.75,
            ),
            symbols=no_meme,
        ),
    ]
    return scenarios


def run_stress_battery(
    cfg: MarketMakingConfig,
    *,
    months: list[tuple[int, int]] | None = None,
    timeframe: str = "1h",
    seed: int = 42,
    initial_equity: float = 1000.0,
    scenarios: list[StressScenario] | None = None,
    deep_loss_pct: float = -5.0,
) -> StressBatteryResult:
    months = months or default_blind_months()
    scenarios = scenarios or build_stress_scenarios(cfg)
    cases: list[StressCaseResult] = []

    for i, sc in enumerate(scenarios):
        trial = sc.mutate(cfg)
        use_months = sc.months or months
        use_seed = seed if sc.seed is None else sc.seed
        use_syms = sc.symbols
        print(f"stress {i + 1}/{len(scenarios)}: {sc.name}", flush=True)
        out = _compound_months(
            trial,
            use_months,
            timeframe=timeframe,
            seed=use_seed,
            initial_equity=initial_equity,
            symbols=use_syms,
        )
        survived = out["final_equity"] > out["start_equity"]
        deep = any(r["ret_pct"] <= deep_loss_pct for r in out["monthly"])
        broken = (not survived) or deep
        cases.append(
            StressCaseResult(
                name=sc.name,
                risk=sc.risk,
                description=sc.description,
                start_equity=out["start_equity"],
                final_equity=out["final_equity"],
                total_return_pct=out["total_return_pct"],
                n_pos_months=out["n_pos_months"],
                n_neg_months=out["n_neg_months"],
                n_months=out["n_months"],
                total_fees=out["total_fees"],
                worst_month=out["worst_month"],
                best_month=out["best_month"],
                monthly=out["monthly"],
                survived=survived,
                broken=broken,
            )
        )

    broken_names = [c.name for c in cases if c.broken]
    return StressBatteryResult(
        baseline_params={
            "gamma": cfg.gamma,
            "kappa": cfg.kappa,
            "tau_bars": cfg.tau_bars,
            "size_eta": cfg.size_eta,
            "adverse_vol_mult": cfg.adverse_vol_mult,
            "vol_lookback": cfg.vol_lookback,
            "ewma_halflife": cfg.ewma_halflife,
            "fill_prob": cfg.sim.fill_prob,
            "adverse_frac": cfg.sim.adverse_frac,
            "commission_pct": cfg.commission_pct,
        },
        window=[f"{y:04d}-{m:02d}" for y, m in months],
        timeframe=timeframe,
        start_equity=float(initial_equity),
        cases=cases,
        n_survived=int(sum(1 for c in cases if c.survived)),
        n_broken=int(len(broken_names)),
        broken_names=broken_names,
        interpretation_allowed=False,
        method=(
            "Compound portfolio stresses on post-calib blind window; "
            "broken = final<=start OR any month <= -5%. Not live edge."
        ),
        notes=(
            "Survived = end equity > start under that stress. "
            "Does not prove live robustness (candle fill model remains)."
        ),
    )


def save_stress_battery(
    result: StressBatteryResult,
    *,
    root: str | Path = "data",
    exchange_id: str = "okx",
    basename: str = "stress_hourly_blind_2026",
) -> dict[str, str]:
    out = Path(root) / exchange_id / "market_making"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{basename}.json"
    path.write_text(json.dumps(asdict(result), indent=2, default=str), encoding="utf-8")
    art = Path("/opt/cursor/artifacts")
    art.mkdir(parents=True, exist_ok=True)
    art_path = art / f"{basename}.json"
    art_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    # slim CSV-like table
    lines = ["name,risk,ret_pct,final,survived,broken,n_pos,n_neg,worst_ret"]
    for c in result.cases:
        lines.append(
            f"{c.name},{c.risk},{c.total_return_pct:.4f},{c.final_equity:.2f},"
            f"{int(c.survived)},{int(c.broken)},{c.n_pos_months},{c.n_neg_months},"
            f"{c.worst_month.get('ret_pct')}"
        )
    table = art / f"{basename}_table.csv"
    table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"summary": str(path), "artifact": str(art_path), "table": str(table)}
