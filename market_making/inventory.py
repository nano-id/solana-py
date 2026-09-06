"""Spot inventory ledger (base + quote cash)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class InventoryState:
    base: float = 0.0
    quote: float = 0.0
    target: float = 0.0
    max_abs: float = 1.0
    halt_frac: float = 0.95
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    n_buys: int = 0
    n_sells: int = 0
    avg_cost: float = 0.0
    events: list[dict] = field(default_factory=list)

    @property
    def deviation(self) -> float:
        return self.base - self.target

    @property
    def utilization(self) -> float:
        return abs(self.deviation) / self.max_abs if self.max_abs > 0 else 0.0

    @property
    def halted(self) -> bool:
        return self.utilization >= self.halt_frac

    def mark_to_market(self, mid: float) -> float:
        return self.quote + self.base * mid

    def apply_buy(self, *, price: float, size: float, fee: float, bar: int) -> None:
        self.quote -= price * size + fee
        if self.base >= -1e-15:
            new_base = self.base + size
            if new_base > 1e-15:
                self.avg_cost = (
                    (self.avg_cost * self.base + price * size) / new_base
                    if self.base > 1e-15
                    else price
                )
            self.base = new_base
        else:
            cover = min(size, -self.base)
            self.realized_pnl += (self.avg_cost - price) * cover
            self.base += size
            if self.base > 1e-15:
                self.avg_cost = price
            elif abs(self.base) <= 1e-15:
                self.base = 0.0
                self.avg_cost = 0.0
        self.fees_paid += fee
        self.n_buys += 1
        self.events.append(
            {"bar": bar, "side": "buy", "price": price, "size": size, "fee": fee}
        )

    def apply_sell(self, *, price: float, size: float, fee: float, bar: int) -> None:
        self.quote += price * size - fee
        if self.base <= 1e-15:
            prev_short = -self.base
            self.base -= size
            new_short = -self.base
            if new_short > 1e-15:
                self.avg_cost = (
                    (self.avg_cost * prev_short + price * size) / new_short
                    if prev_short > 1e-15
                    else price
                )
        else:
            sold = min(size, self.base)
            self.realized_pnl += (price - self.avg_cost) * sold
            self.base -= size
            if abs(self.base) <= 1e-15:
                self.base = 0.0
                self.avg_cost = 0.0
            elif self.base < 0:
                self.avg_cost = price
        self.fees_paid += fee
        self.n_sells += 1
        self.events.append(
            {"bar": bar, "side": "sell", "price": price, "size": size, "fee": fee}
        )
