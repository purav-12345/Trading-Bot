"""The risk engine — the single most important module in this project.

It is deterministic, fast, and has the FINAL say. Strategies and the AI
agent can only *propose*; nothing reaches the broker without passing here.

Two jobs:
  1) Per-order gate  -> approve/resize/reject a proposed order.
  2) Daily guardrail -> on each tick, check total P&L against the profit
     target and max-loss. When breached: flatten everything, lock the day,
     and refuse all further trading until a manual reset.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..brokers.base import Broker
    from ..config import Config
    from ..store.db import Store


@dataclass
class RiskDecision:
    allowed: bool
    quantity: int          # possibly resized down from the request
    reason: str = ""


class LimitHit(Exception):
    """Raised internally when a daily limit triggers the kill switch."""


class RiskEngine:
    def __init__(self, config: "Config", broker: "Broker", store: "Store"):
        self.config = config
        self.broker = broker
        self.store = store
        r = config.risk
        self.profit_target = float(r.get("daily_profit_target", 30000))
        self.max_loss = float(r.get("daily_max_loss", 2000))
        self.max_trades = int(r.get("max_trades_per_day", 50))
        self.max_open = int(r.get("max_open_positions", 5))
        self.per_trade_risk = float(r.get("per_trade_risk_rupees", 1000))
        self.max_capital_per_trade = float(r.get("max_capital_per_trade", 50000))
        self.square_off_on_limit = bool(r.get("square_off_on_limit", True))
        self.lock_until_reset = bool(r.get("lock_until_manual_reset", True))
        # Optional NET-WORTH kill switch (HAASEY-style: floor/ceiling on total
        # portfolio value = starting capital + P&L). None = disabled.
        self.starting_capital = float(
            r.get("starting_capital", config.account.get("paper_starting_cash", 100000))
        )
        self.capital_floor = r.get("capital_floor")      # e.g. 15000 -> stop if worth <= 15k
        self.capital_ceiling = r.get("capital_ceiling")  # e.g. 25000 -> stop if worth >= 25k

    def net_worth(self) -> float:
        return self.starting_capital + self.total_pnl()

    # ------------------------------------------------------------------
    # Daily P&L = realized (persisted) + live unrealized (from broker).
    # ------------------------------------------------------------------
    def realized_pnl(self) -> float:
        return float(self.store.get_day()["realized_pnl"])

    def unrealized_pnl(self) -> float:
        try:
            return float(self.broker.unrealized_pnl())
        except Exception:
            return 0.0

    def total_pnl(self) -> float:
        return self.realized_pnl() + self.unrealized_pnl()

    def is_locked(self) -> bool:
        return self.store.is_locked()

    # ------------------------------------------------------------------
    # Per-order gate.
    # ------------------------------------------------------------------
    def evaluate_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        is_exit: bool = False,
    ) -> RiskDecision:
        # Exits (square-offs / stop-losses / target sells) are ALWAYS allowed:
        # reducing risk must never be blocked.
        if is_exit:
            return RiskDecision(True, quantity, "exit allowed")

        if self.is_locked():
            return RiskDecision(False, 0, "day is locked (limit hit / manual stop)")

        day = self.store.get_day()
        if day["trade_count"] >= self.max_trades:
            return RiskDecision(False, 0, f"max trades/day reached ({self.max_trades})")

        open_positions = sum(1 for p in self.broker.positions().values() if not p.is_flat)
        if open_positions >= self.max_open and self.broker.positions().get(symbol, None) is None:
            return RiskDecision(False, 0, f"max open positions reached ({self.max_open})")

        if price <= 0:
            return RiskDecision(False, 0, "no price")

        # Cap capital per trade -> resize quantity down if needed.
        qty = quantity
        max_qty_by_capital = int(self.max_capital_per_trade // price)
        if max_qty_by_capital < 1:
            return RiskDecision(False, 0, "single share exceeds per-trade capital cap")
        if qty > max_qty_by_capital:
            qty = max_qty_by_capital

        # Don't open a position whose worst-case (to the day's remaining loss
        # budget) would blow the daily max-loss in one shot.
        remaining_loss_budget = self.max_loss + self.total_pnl()  # how far from the floor
        if remaining_loss_budget <= 0:
            return RiskDecision(False, 0, "no loss budget remaining")

        if qty < 1:
            return RiskDecision(False, 0, "resized below 1 share")
        note = "ok" if qty == quantity else f"resized {quantity}->{qty} (capital cap)"
        return RiskDecision(True, qty, note)

    # ------------------------------------------------------------------
    # Daily guardrail — call this every tick. Returns True if still tradeable.
    # ------------------------------------------------------------------
    def check_daily_limits(self, notifier=None) -> bool:
        if self.is_locked():
            return False

        pnl = self.total_pnl()
        if pnl >= self.profit_target:
            self._trigger(f"PROFIT TARGET hit: P&L Rs.{pnl:,.0f} >= Rs.{self.profit_target:,.0f}", notifier)
            return False
        if pnl <= -self.max_loss:
            self._trigger(f"MAX LOSS hit: P&L Rs.{pnl:,.0f} <= -Rs.{self.max_loss:,.0f}", notifier)
            return False
        # Net-worth kill switch (HAASEY): portfolio value floor/ceiling.
        if self.capital_floor is not None or self.capital_ceiling is not None:
            worth = self.net_worth()
            if self.capital_floor is not None and worth <= float(self.capital_floor):
                self._trigger(f"CAPITAL FLOOR hit: worth Rs.{worth:,.0f} <= Rs.{float(self.capital_floor):,.0f}", notifier)
                return False
            if self.capital_ceiling is not None and worth >= float(self.capital_ceiling):
                self._trigger(f"CAPITAL CEILING hit: worth Rs.{worth:,.0f} >= Rs.{float(self.capital_ceiling):,.0f}", notifier)
                return False
        return True

    def _trigger(self, reason: str, notifier=None) -> None:
        self.store.audit("LIMIT_TRIGGERED", reason)
        # 1) Flatten everything (book the open P&L into realized).
        if self.square_off_on_limit:
            results = self.broker.square_off_all()
            booked = self.broker.realized_pnl()
            # Persist realized so the lock survives a restart.
            self.store.update_day(realized_pnl=booked)
            self.store.audit("SQUARE_OFF", [r.__dict__ for r in results])
        # 2) Lock the day.
        if self.lock_until_reset:
            self.store.lock_day(reason)
        if notifier:
            try:
                notifier.send(f"🛑 Trading stopped for the day.\n{reason}\n"
                              f"Locked until you press Run / reset.")
            except Exception:
                pass

    def manual_stop(self, reason: str = "manual stop", notifier=None) -> None:
        self._trigger(reason, notifier)

    def reset_day(self) -> None:
        """User pressed Run again / made a new algo: clear the lock."""
        self.store.unlock_day()
