"""Funding-rate arbitrage engine (Module 1 live collector + later execution).

Step 1: live market snapshot collection (funding, spot, perp, order-book depth).
Later steps: screen, delta-neutral entry, rotation, exits, trade log.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
