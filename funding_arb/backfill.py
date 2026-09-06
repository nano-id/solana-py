"""Multi-day observation backfill from OKX funding settlements.

OKX public funding history only retains ~3 months. If the requested calendar
window (e.g. Feb 5–15) is older than that, we map to the same day-of-month
window in the earliest available month and document the substitution.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from funding_arb.config import FundingArbAppConfig, load_funding_arb_config
from funding_arb.control import run_control_cycle
from funding_arb.journal import JournalCosts, record_cycle
from funding_arb.position import PaperPosition
from funding_arb.screen import screen_opportunities
from funding_arb.storage import append_history, snapshots_dir

logger = logging.getLogger(__name__)

DEFAULT_INTEREST_RATE = 0.0001
FLOOR_EPS = 1e-9


def _parse_date(value: str) -> datetime:
    raw = value.strip()
    if len(raw) == 10:
        raw = f"{raw}T00:00:00"
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _symbol_to_funding_path(root: Path, symbol: str) -> Path:
    stem = symbol.replace("/", "_").replace(":", "_")
    return Path(root) / "okx" / "funding" / f"{stem}.parquet"


def load_local_funding_history(
    symbols: list[str],
    *,
    data_root: Path | str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    root = Path(data_root)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    chunks: list[pd.DataFrame] = []
    for sym in symbols:
        path = _symbol_to_funding_path(root, sym)
        if not path.is_file():
            logger.warning("missing local funding file for %s (%s)", sym, path)
            continue
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        ts = pd.to_numeric(frame["timestamp"], errors="coerce")
        hit = frame[(ts >= start_ms) & (ts < end_ms)].copy()
        if hit.empty:
            continue
        hit["symbol"] = sym
        hit["funding_rate"] = pd.to_numeric(hit["funding_rate"], errors="coerce")
        chunks.append(hit)
    if not chunks:
        return pd.DataFrame(columns=["timestamp", "funding_rate", "symbol"])
    return (
        pd.concat(chunks, ignore_index=True)
        .sort_values(["timestamp", "symbol"])
        .reset_index(drop=True)
    )


def fetch_okx_funding_history(
    symbols: list[str],
    *,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    from pairs_trading.data.config import ExchangeConfig
    from pairs_trading.data.exchange import connect_with_fallback
    from pairs_trading.funding.fetcher import fetch_funding_rate_history

    connected = connect_with_fallback(
        ExchangeConfig(id="okx", fallbacks=[], market_type="swap")
    )
    since_ms = int(start.timestamp() * 1000)
    until_ms = int(end.timestamp() * 1000)
    chunks: list[pd.DataFrame] = []
    for sym in symbols:
        try:
            frame = fetch_funding_rate_history(
                connected.exchange,
                sym,
                since_ms=since_ms,
                until_ms=until_ms,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("history fetch failed for %s: %s", sym, exc)
            continue
        if frame.empty:
            continue
        frame = frame.copy()
        frame["symbol"] = sym
        chunks.append(frame)
    if not chunks:
        return pd.DataFrame(columns=["timestamp", "funding_rate", "symbol"])
    return (
        pd.concat(chunks, ignore_index=True)
        .sort_values(["timestamp", "symbol"])
        .reset_index(drop=True)
    )


def depth_lookup_from_latest(root: Path | str, exchange_id: str) -> dict[str, float]:
    path = snapshots_dir(root, exchange_id) / "latest.parquet"
    if not path.is_file():
        return {}
    latest = pd.read_parquet(path)
    if latest.empty or "symbol" not in latest.columns or "depth_usdt" not in latest.columns:
        return {}
    out: dict[str, float] = {}
    for _, row in latest.iterrows():
        try:
            out[str(row["symbol"])] = float(row["depth_usdt"])
        except (TypeError, ValueError):
            continue
    return out


def funding_rows_to_snapshots(
    funding: pd.DataFrame,
    *,
    exchange_id: str = "okx",
    interest_rate: float = DEFAULT_INTEREST_RATE,
    depth_by_symbol: dict[str, float] | None = None,
    default_depth_usdt: float = 50_000.0,
) -> pd.DataFrame:
    if funding.empty:
        return pd.DataFrame()
    depth_by_symbol = depth_by_symbol or {}
    rows: list[dict[str, Any]] = []
    for _, fr in funding.iterrows():
        if pd.isna(fr["funding_rate"]):
            continue
        rate = float(fr["funding_rate"])
        ts = int(fr["timestamp"])
        observed_at = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).isoformat()
        symbol = str(fr["symbol"])
        at_floor = abs(rate - float(interest_rate)) < FLOOR_EPS
        depth = float(depth_by_symbol.get(symbol, default_depth_usdt))
        base = symbol.split(":")[0] if ":" in symbol else symbol
        spot_symbol = base if "/" in base else f"{base}/USDT"
        rows.append(
            {
                "observed_at": observed_at,
                "exchange_id": exchange_id,
                "symbol": symbol,
                "spot_symbol": spot_symbol,
                "funding_rate": rate,
                "funding_rate_raw": f"{rate:.16g}",
                "sett_funding_rate": rate,
                "sett_funding_rate_raw": f"{rate:.16g}",
                "interest_rate": float(interest_rate),
                "interest_rate_raw": f"{interest_rate:.16g}",
                "at_interest_floor": bool(at_floor),
                "mark_price": None,
                "index_price": None,
                "perp_last": None,
                "spot_last": None,
                "quote_volume_24h": None,
                "basis_bps_spot": None,
                "perp_contract_size": None,
                "perp_spread_bps": None,
                "perp_bid_notional": None,
                "perp_ask_notional": None,
                "perp_min_notional": None,
                "spot_spread_bps": None,
                "spot_bid_notional": None,
                "spot_ask_notional": None,
                "spot_min_notional": None,
                "depth_usdt": depth,
                "depth_band_frac": 0.002,
                "backfill": True,
            }
        )
    return pd.DataFrame(rows)


def resolve_observation_window(
    requested_start: str,
    requested_end: str,
    *,
    available_start: datetime | None = None,
) -> tuple[datetime, datetime, dict[str, Any]]:
    start = _parse_date(requested_start)
    end = _parse_date(requested_end)
    meta: dict[str, Any] = {
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "substituted": False,
        "reason": "ok",
    }
    if available_start is None or start >= available_start:
        return start, end, meta

    if end > available_start:
        meta["substituted"] = True
        meta["reason"] = "clamped_start_to_available"
        return available_start, end, meta

    month_anchor = available_start.replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    )
    try:
        alt_start = month_anchor.replace(day=start.day)
        alt_end = month_anchor.replace(day=end.day)
    except ValueError:
        alt_start = available_start
        alt_end = available_start + timedelta(days=max(1, (end - start).days))
    if alt_start < available_start:
        alt_start = available_start
    if alt_end <= alt_start:
        alt_end = alt_start + timedelta(days=10)
    meta["substituted"] = True
    meta["reason"] = (
        f"OKX history starts {available_start.date()}; "
        f"mapped {start.date()}→{end.date()} to {alt_start.date()}→{alt_end.date()}"
    )
    return alt_start, alt_end, meta


def write_observation_history(
    snapshots: pd.DataFrame,
    *,
    root: Path | str,
    exchange_id: str,
    replace: bool = True,
) -> Path:
    base = snapshots_dir(root, exchange_id)
    base.mkdir(parents=True, exist_ok=True)
    path = base / "history.parquet"
    if replace or not path.is_file():
        snapshots.to_parquet(path, index=False)
    else:
        append_history(snapshots, path)
    meta = {
        "rows": int(len(snapshots)),
        "symbols": int(snapshots["symbol"].nunique()) if not snapshots.empty else 0,
        "first_ts": str(snapshots["observed_at"].min()) if not snapshots.empty else None,
        "last_ts": str(snapshots["observed_at"].max()) if not snapshots.empty else None,
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    (base / "observation_backfill.meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return path


def replay_paper_cycles(
    snapshots: pd.DataFrame,
    *,
    config: FundingArbAppConfig,
    exchange_id: str,
    clear_journal: bool = True,
) -> dict[str, Any]:
    root = Path(config.storage.root)
    journal_base = root / exchange_id / "funding_arb" / "journal"
    if clear_journal and journal_base.exists():
        for p in journal_base.glob("trades.*"):
            p.unlink(missing_ok=True)

    times = sorted(snapshots["observed_at"].astype(str).unique())
    position: PaperPosition | None = None
    actions: dict[str, int] = {}
    history_so_far = pd.DataFrame()

    for ts in times:
        snap = snapshots[snapshots["observed_at"].astype(str) == ts].reset_index(drop=True)
        history_so_far = pd.concat([history_so_far, snap], ignore_index=True)
        screen = screen_opportunities(
            snap,
            settlements_per_day=config.collector.settlements_per_day,
            min_depth_usdt=config.screen.min_depth_usdt,
            rate_col=config.screen.rate_col,
            history=history_so_far,
            floor_lookback=config.screen.floor_lookback,
            prefer_non_floor=config.screen.prefer_non_floor,
        )
        result = run_control_cycle(
            screen,
            position,
            exchange_id=exchange_id,
            min_edge_multiple=config.rotation.min_edge_multiple,
            rotation_cost=config.rotation.rotation_cost,
            require_non_floor=config.rotation.require_non_floor,
            exit_on_floor=config.exit.exit_on_floor,
            exit_on_adverse_funding=config.exit.exit_on_adverse_funding,
            depth_fraction=config.entry.depth_fraction,
            max_notional_usdt=config.entry.max_notional_usdt,
            min_notional_usdt=config.entry.min_notional_usdt,
            flatten_unpaired=config.entry.flatten_unpaired,
        )
        if result.position_after is not None and result.action.value in {
            "enter",
            "rotate",
        }:
            result.position_after.opened_at = str(ts)

        if config.journal.enabled:
            record_cycle(
                result,
                root=root,
                exchange_id=exchange_id,
                costs=JournalCosts(
                    rotation_cost=config.journal.rotation_cost,
                    slippage_cost=config.journal.slippage_cost,
                ),
                settlements_per_day=config.collector.settlements_per_day,
            )
        actions[result.action.value] = actions.get(result.action.value, 0) + 1
        position = result.position_after

    pos_path = root / exchange_id / "funding_arb" / "paper_position.json"
    pos_path.parent.mkdir(parents=True, exist_ok=True)
    if position is not None:
        pos_path.write_text(
            json.dumps(position.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
    elif pos_path.is_file():
        pos_path.unlink()

    return {
        "settlement_times": int(len(times)),
        "actions": actions,
        "final_position": position.to_dict() if position else None,
    }


def run_observation_backfill(
    *,
    start: str,
    end: str,
    config: FundingArbAppConfig | None = None,
    exchange_id: str = "okx",
    source: str = "local",
    replace_history: bool = True,
    replay: bool = True,
) -> dict[str, Any]:
    config = config or load_funding_arb_config()
    symbols = list(config.collector.symbols)

    probe = load_local_funding_history(
        symbols[:3],
        data_root=config.storage.root,
        start=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end=datetime(2099, 1, 1, tzinfo=timezone.utc),
    )
    available_start = None
    if not probe.empty:
        available_start = datetime.fromtimestamp(
            int(probe["timestamp"].min()) / 1000.0, tz=timezone.utc
        )

    eff_start, eff_end, window_meta = resolve_observation_window(
        start, end, available_start=available_start
    )

    if source == "live":
        funding = fetch_okx_funding_history(symbols, start=eff_start, end=eff_end)
    else:
        funding = load_local_funding_history(
            symbols, data_root=config.storage.root, start=eff_start, end=eff_end
        )
        if funding.empty:
            logger.info("local funding empty for window; falling back to live fetch")
            funding = fetch_okx_funding_history(symbols, start=eff_start, end=eff_end)

    depths = depth_lookup_from_latest(config.storage.root, exchange_id)
    snapshots = funding_rows_to_snapshots(
        funding,
        exchange_id=exchange_id,
        depth_by_symbol=depths,
    )
    hist_path = write_observation_history(
        snapshots,
        root=config.storage.root,
        exchange_id=exchange_id,
        replace=replace_history,
    )

    replay_stats: dict[str, Any] = {}
    if replay and not snapshots.empty:
        replay_stats = replay_paper_cycles(
            snapshots, config=config, exchange_id=exchange_id, clear_journal=True
        )

    return {
        "window": window_meta,
        "effective_start": eff_start.isoformat(),
        "effective_end": eff_end.isoformat(),
        "funding_rows": int(len(funding)),
        "snapshot_rows": int(len(snapshots)),
        "n_symbols": int(snapshots["symbol"].nunique()) if not snapshots.empty else 0,
        "n_settlement_times": (
            int(snapshots["observed_at"].nunique()) if not snapshots.empty else 0
        ),
        "history_path": str(hist_path),
        "replay": replay_stats,
        "source": source if not funding.empty else "empty",
    }
