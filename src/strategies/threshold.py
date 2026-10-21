"""SCENARIO 2 — the live buy/sell/buy cycle (your GAIL example).

Logic (long-only, intraday):
  - Flat  & LTP <= entry_price        -> BUY  `quantity`
  - Long  & LTP >= target_price       -> SELL (book profit)   [exit]
  - Long  & LTP <= stoploss_price     -> SELL (cut loss)      [exit]
  - After exiting, if re_enter is true, it waits to BUY again
    when price comes back to entry_price -> keeps cycling all day.
"""
from __future__ import annotations

from .base import Signal, Strategy, StrategyContext


class ThresholdStrategy(Strategy):
    type = "threshold"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.entry_price = float(cfg["entry_price"])
        self.target_price = float(cfg["target_price"])
        self.stoploss_price = float(cfg.get("stoploss_price", 0) or 0)
        self.re_enter = bool(cfg.get("re_enter", True))
        self._done = False  # set when re_enter is False and one cycle completed

    def on_tick(self, ctx: StrategyContext) -> list[Signal]:
        if self._done:
            return []
        ltp = ctx.ltp(self.symbol)
        pos = ctx.position(self.symbol)

        # Holding a long -> look to exit.
        if pos.quantity > 0:
            if ltp >= self.target_price:
                return [self._exit(pos.quantity, f"target {self.target_price} hit @ {ltp}")]
            if self.stoploss_price and ltp <= self.stoploss_price:
                return [self._exit(pos.quantity, f"stoploss {self.stoploss_price} hit @ {ltp}")]
            return []

        # Flat -> look to enter.
        if ltp <= self.entry_price:
            return [Signal(self.symbol, "BUY", self.quantity,
                           reason=f"entry {self.entry_price} reached @ {ltp}")]
        return []

    def _exit(self, qty: int, reason: str) -> Signal:
        if not self.re_enter:
            self._done = True
        return Signal(self.symbol, "SELL", qty, reason=reason, is_exit=True)
