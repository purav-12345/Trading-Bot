"""Broker abstraction. Paper and Zerodha both implement this, so the rest
of the bot never knows or cares whether money is real."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OrderResult:
    ok: bool
    order_id: str
    symbol: str
    side: str            # BUY / SELL
    quantity: int
    fill_price: float
    status: str          # FILLED / REJECTED / PENDING
    message: str = ""


@dataclass
class Position:
    symbol: str
    quantity: int = 0          # signed: +long / -short
    avg_price: float = 0.0
    last_price: float = 0.0
    realized_pnl: float = 0.0

    @property
    def unrealized_pnl(self) -> float:
        return (self.last_price - self.avg_price) * self.quantity

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0


class Broker(ABC):
    name: str = "base"

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def ltp(self, symbol: str) -> float: ...

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = "MARKET",
        price: Optional[float] = None,
    ) -> OrderResult: ...

    @abstractmethod
    def positions(self) -> dict[str, Position]: ...

    @abstractmethod
    def square_off_all(self) -> list[OrderResult]: ...

    # ---- shared P&L helpers (work off positions) ----
    def unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions().values())

    def realized_pnl(self) -> float:
        return sum(p.realized_pnl for p in self.positions().values())

    def total_pnl(self) -> float:
        return self.realized_pnl() + self.unrealized_pnl()
