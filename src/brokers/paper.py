"""Paper broker: simulates fills at the live LTP from the market-data feed.

This is the DEFAULT. It is the real engine end-to-end (same code path as
live), but orders are simulated so you can watch behaviour before risking
a rupee. P&L accounting is FIFO-ish via running average cost.
"""
from __future__ import annotations

import uuid
from typing import Optional

from .base import Broker, OrderResult, Position


class PaperBroker(Broker):
    name = "paper"

    def __init__(self, config, market_data):
        self.config = config
        self.md = market_data
        self.cash = float(config.account.get("paper_starting_cash", 100000))
        self._positions: dict[str, Position] = {}

    def connect(self) -> None:
        self.md.connect()

    def ltp(self, symbol: str) -> float:
        return self.md.ltp(symbol)

    def _pos(self, symbol: str) -> Position:
        return self._positions.setdefault(symbol, Position(symbol=symbol))

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = "MARKET",
        price: Optional[float] = None,
    ) -> OrderResult:
        side = side.upper()
        if quantity <= 0:
            return OrderResult(False, "", symbol, side, quantity, 0.0, "REJECTED", "qty<=0")

        ltp = self.ltp(symbol)
        # LIMIT orders only fill if the market is at/through the limit.
        fill = ltp
        if order_type.upper() == "LIMIT" and price is not None:
            if side == "BUY" and ltp > price:
                return OrderResult(False, "", symbol, side, quantity, 0.0, "PENDING", "limit not reached")
            if side == "SELL" and ltp < price:
                return OrderResult(False, "", symbol, side, quantity, 0.0, "PENDING", "limit not reached")
            fill = price

        pos = self._pos(symbol)
        signed = quantity if side == "BUY" else -quantity
        new_qty = pos.quantity + signed

        # Realize P&L when reducing/closing an existing position.
        if pos.quantity != 0 and (pos.quantity > 0) != (signed > 0):
            closing = min(abs(signed), abs(pos.quantity))
            direction = 1 if pos.quantity > 0 else -1
            pos.realized_pnl += (fill - pos.avg_price) * closing * direction

        if new_qty == 0:
            pos.avg_price = 0.0
        elif (pos.quantity >= 0) == (signed > 0) or pos.quantity == 0:
            # adding in the same direction -> blend average cost
            total_cost = pos.avg_price * abs(pos.quantity) + fill * abs(signed)
            pos.avg_price = total_cost / abs(new_qty)
        # (flip-through keeps the new entry price set below)
        if (pos.quantity > 0) != (new_qty > 0) and new_qty != 0:
            pos.avg_price = fill

        pos.quantity = new_qty
        pos.last_price = ltp
        self.cash -= fill * signed
        return OrderResult(
            True, f"paper-{uuid.uuid4().hex[:10]}", symbol, side, quantity, fill, "FILLED"
        )

    def positions(self) -> dict[str, Position]:
        for sym, pos in self._positions.items():
            try:
                pos.last_price = self.ltp(sym)
            except Exception:
                pass
        return self._positions

    def square_off_all(self) -> list[OrderResult]:
        results = []
        for sym, pos in list(self._positions.items()):
            if pos.quantity != 0:
                side = "SELL" if pos.quantity > 0 else "BUY"
                results.append(self.place_order(sym, side, abs(pos.quantity), "MARKET"))
        return results
