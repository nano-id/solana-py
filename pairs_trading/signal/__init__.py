"""Phase 3 signal package: z-score skeleton tested on synthetic OU spreads."""

from pairs_trading.signal.engine import SignalConfig, SignalEvent, Side, generate_signals, summarize_signals
from pairs_trading.signal.ou import simulate_ou_spread

__all__ = [
    "Side",
    "SignalConfig",
    "SignalEvent",
    "generate_signals",
    "simulate_ou_spread",
    "summarize_signals",
]
