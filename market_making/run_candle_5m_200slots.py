"""$200 dual-slot paper bot — 5m candle clock (decision @ 4:59).

Decision rule
-------------
Candles are 5 minutes. The decision price for each bar is the price at
second 299 of that bar (4m59s), i.e. the 5m candle close on historical data.

This is the same AS dual-slot engine as the retail 5m forward runner, with
fixed $100 windows (2 slots from a $200 start). No monthly DCA.

Live counterpart: ``run_live_paper_200slots`` gates on wall-clock second 299.
interpretation_allowed=false.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from market_making.run_retail_5m_250slots import main as retail_main

START_EQUITY = 200.0
SLOT = 100.0
REMNANT_MIN = 100.0
DECISION_SEC_IN_BAR = 299  # 4m59s within each 5m candle
BAR_SEC = 300


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="$200 dual-slot 5m candle paper (decision @ 4:59 / close)"
    )
    p.add_argument("--config", default="config/market_making_5m_amj2026.yaml")
    p.add_argument("--start", default="2026-07-01")
    p.add_argument("--end-exclusive", default="2026-09-07")
    p.add_argument("--start-equity", type=float, default=START_EQUITY)
    p.add_argument("--slot", type=float, default=SLOT)
    p.add_argument("--remnant-min", type=float, default=REMNANT_MIN)
    p.add_argument("--out-basename", default="")
    args = p.parse_args(argv)

    slot = float(args.slot)
    equity = float(args.start_equity)
    remnant = float(args.remnant_min)
    basename = args.out_basename.strip() or (
        f"candle_5m_{int(equity)}_slot{int(slot)}_"
        f"{args.start[:7].replace('-', '')}_{args.end_exclusive[:7].replace('-', '')}"
    )

    # Delegate to the shared 5m dual-slot engine (one decision per 5m close).
    code = retail_main(
        [
            "--config",
            args.config,
            "--start",
            args.start,
            "--end-exclusive",
            args.end_exclusive,
            "--start-equity",
            str(equity),
            "--monthly-deposit",
            "0",
            "--slot-min",
            str(slot),
            "--slot-max",
            str(slot),
            "--remnant-min",
            str(remnant),
            "--out-basename",
            basename,
        ]
    )

    # Stamp explicit candle-clock metadata onto the results JSON.
    path = Path(f"config/{basename}_results.json")
    if path.exists():
        payload = json.loads(path.read_text())
        design = payload.setdefault("design", {})
        design["decision_clock"] = {
            "timeframe": "5m",
            "bar_seconds": BAR_SEC,
            "decision_second_in_bar": DECISION_SEC_IN_BAR,
            "decision_label": "4m59s",
            "hist_proxy": "5m candle close (== price at end of bar / 4:59)",
            "live_rule": (
                "act once per bar when wall-clock sec_in_bar >= 299 "
                "and bar_id not yet decided"
            ),
        }
        design["params_note"] = (
            "AMJ2026 lock when using market_making_5m_amj2026.yaml; "
            "JFM2025 lock when using market_making_5m.yaml"
        )
        path.write_text(json.dumps(payload, indent=2, default=str))
        art = Path("/opt/cursor/artifacts") / f"{basename}_results.json"
        if art.parent.exists():
            art.write_text(json.dumps(payload, indent=2, default=str))
        print(
            f"Candle clock stamped: decision @ second {DECISION_SEC_IN_BAR}/300 "
            f"(5m close)",
            flush=True,
        )
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
