"""The orchestrator. Wires market data -> strategies -> RISK ENGINE -> broker.

Golden rule enforced here: every proposed entry passes through
`risk.evaluate_order`; every tick calls `risk.check_daily_limits`. Exits are
always allowed. The LLM is never in this loop except as a cached, capped
veto inside a strategy that opted in.
"""
from __future__ import annotations

import time
from datetime import datetime, time as dtime
from typing import Optional

import pytz

from ..ai.gemini import GeminiHelper
from ..brokers import make_broker
from ..config import Config
from ..market import MarketData
from ..notify import Notifier
from ..risk import RiskEngine
from ..store.db import Store
from ..strategies import Signal, StrategyContext, build_strategy


def _parse_hhmm(s: str) -> dtime:
    h, m = s.split(":")
    return dtime(int(h), int(m))


class Trader:
    def __init__(self, config: Config):
        self.config = config
        self.tz = pytz.timezone(config.session.get("timezone", "Asia/Kolkata"))
        self.store = Store()
        self.md = MarketData(config)
        self.broker = make_broker(config, self.md)
        self.risk = RiskEngine(config, self.broker, self.store)
        self.notifier = Notifier(config)
        self.ai = GeminiHelper(config) if config.ai.get("enabled") else None
        self.strategies = [build_strategy(c) for c in config.enabled_strategies()]

        sess = config.session
        start = _parse_hhmm(sess.get("market_open", "09:15"))
        self.ctx = StrategyContext(self.broker, session_start=start, ai=self.ai)
        self._eod_sent = False
        syms: set[str] = set()
        for s in self.strategies:
            syms.add(s.symbol)
            syms.update(getattr(s, "universe", []))  # HAASEY scans a whole universe
            if getattr(s, "use_regime", False):
                syms.add(s.regime_symbol)             # index bars for the regime filter
        self._symbols = sorted(syms)
        self._bar_state: dict[str, dict] = {}

    # ------------------------------------------------------------------
    def connect(self) -> None:
        self.broker.connect()
        mode = "LIVE 💸" if self.config.secrets.is_live else "PAPER (simulated)"
        data = "live Kite data" if self.md.using_live_data else "SYNTHETIC data"
        self.store.audit("CONNECT", {"mode": mode, "data": data})
        print(f"[trader] connected | mode={mode} | {data} | strategies={[s.name for s in self.strategies]}")
        self._seed_bars()

    def now(self) -> datetime:
        return datetime.now(self.tz)

    # ------------------------------------------------------------------
    # Bar handling: warm-start from history, then aggregate live ticks into
    # 1-minute OHLC bars so VWAP/RSI/ATR/Supertrend strategies work live.
    # (Live volume is unavailable without a Kite data plan, so VWAP degrades
    # to a typical-price mean — documented, still functional.)
    # ------------------------------------------------------------------
    def _seed_bars(self) -> None:
        # Enough history to warm up 15-min-bar indicators (SMA50 etc.):
        # ~5 days of 1-min bars -> plenty of decision bars after resampling.
        for sym in self._symbols:
            try:
                df = self.md.historical(sym, interval="minute", days=5)
            except Exception:
                continue
            for _, row in df.tail(2200).iterrows():
                self.ctx.record_bar(sym, {
                    "date": row.get("date"), "open": row.get("open"),
                    "high": row.get("high"), "low": row.get("low"),
                    "close": row.get("close"), "volume": row.get("volume", 0.0),
                })

    def _update_bars(self) -> None:
        minute = self.now().replace(second=0, microsecond=0)
        for sym in self._symbols:
            try:
                price = self.broker.ltp(sym)
            except Exception:
                continue
            st = self._bar_state.get(sym)
            if st is None or st["minute"] != minute:
                if st is not None:
                    self.ctx.record_bar(sym, st["bar"])  # close the completed minute
                self._bar_state[sym] = {
                    "minute": minute,
                    "bar": {"date": minute, "open": price, "high": price,
                            "low": price, "close": price, "volume": 0.0},
                }
            else:
                bar = st["bar"]
                bar["high"] = max(bar["high"], price)
                bar["low"] = min(bar["low"], price)
                bar["close"] = price

    # ------------------------------------------------------------------
    # One pass over all strategies.
    # ------------------------------------------------------------------
    def run_once(self, allow_new: bool = True) -> None:
        self.ctx.now = self.now()
        self._update_bars()
        for strat in self.strategies:
            try:
                signals = strat.on_tick(self.ctx)
            except Exception as exc:  # noqa: BLE001 - one bad strategy can't kill the loop
                self.store.audit("STRATEGY_ERROR", {"strategy": strat.name, "error": str(exc)})
                continue
            for sig in signals:
                self._handle_signal(strat.name, sig, allow_new)

        # Keep persisted realized P&L in sync with the broker, then guard.
        self.store.update_day(realized_pnl=float(self.broker.realized_pnl()))
        self.risk.check_daily_limits(self.notifier)

    def _handle_signal(self, strategy: str, sig: Signal, allow_new: bool) -> None:
        price = sig.price or self.broker.ltp(sig.symbol)

        if sig.is_exit:
            qty = sig.quantity
        else:
            if not allow_new:
                return  # past the new-entry cutoff: only exits from here
            decision = self.risk.evaluate_order(sig.symbol, sig.side, sig.quantity, price)
            if not decision.allowed:
                self.store.audit("ORDER_BLOCKED", {"strategy": strategy, "symbol": sig.symbol,
                                                   "reason": decision.reason})
                return
            qty = decision.quantity

        result = self.broker.place_order(sig.symbol, sig.side, qty, sig.order_type, sig.price)
        self.store.record_order(
            strategy=strategy, symbol=sig.symbol, side=sig.side, quantity=qty,
            order_type=sig.order_type, price=sig.price, fill_price=result.fill_price,
            status=result.status, broker_order_id=result.order_id,
            mode=self.broker.name, reason=sig.reason,
        )
        if result.ok and result.status == "FILLED":
            self.store.incr_trades()
            print(f"[order] {sig.side} {qty} {sig.symbol} @ {result.fill_price} "
                  f"({strategy}: {sig.reason})")

    # ------------------------------------------------------------------
    # Session loop. respect_session=False lets you paper-test any time.
    # ------------------------------------------------------------------
    def run_session(
        self,
        poll_seconds: float = 2.0,
        respect_session: bool = True,
        max_seconds: Optional[float] = None,
    ) -> None:
        self.connect()
        if self.risk.is_locked():
            reason = self.store.get_day()["lock_reason"]
            msg = f"Day is LOCKED ({reason}). Reset with `python -m src.main reset` to trade again."
            print(f"[trader] {msg}")
            self.notifier.send("ℹ️ " + msg)
            return

        sess = self.config.session
        t_start = _parse_hhmm(sess.get("start_trading", "09:20"))
        t_stop_new = _parse_hhmm(sess.get("stop_new_trades", "15:10"))
        t_squareoff = _parse_hhmm(sess.get("square_off_all", "15:20"))
        started = time.time()

        print(f"[trader] session loop started (poll={poll_seconds}s, respect_session={respect_session})")
        while True:
            if max_seconds and (time.time() - started) >= max_seconds:
                print("[trader] max_seconds reached, stopping loop.")
                break

            if respect_session:
                now_t = self.now().time()
                if now_t >= t_squareoff:
                    self.square_off_eod("end-of-day square-off")
                    break
                if now_t < t_start:
                    time.sleep(poll_seconds)
                    continue
                allow_new = now_t < t_stop_new
            else:
                allow_new = True

            self.run_once(allow_new=allow_new)
            if self.risk.is_locked():
                print("[trader] limit hit -> day locked. Stopping loop.")
                break
            time.sleep(poll_seconds)

    # ------------------------------------------------------------------
    def square_off_eod(self, reason: str) -> None:
        self.broker.square_off_all()
        self.store.update_day(realized_pnl=float(self.broker.realized_pnl()))
        self.store.audit("EOD_SQUARE_OFF", reason)
        self._send_eod_summary()

    def _send_eod_summary(self) -> None:
        if self._eod_sent:
            return
        self._eod_sent = True
        stats = self.status()
        if self.ai and self.config.ai.get("eod_summary"):
            text = self.ai.eod_summary(stats)
        else:
            text = (f"EOD: P&L Rs.{stats['total_pnl']:,.0f} | trades {stats['trade_count']} | "
                    f"status {stats['status']}")
        self.notifier.send("📊 " + text)

    # ------------------------------------------------------------------
    def status(self) -> dict:
        day = self.store.get_day()
        positions = {
            s: {"qty": p.quantity, "avg": round(p.avg_price, 2),
                "ltp": round(p.last_price, 2), "uPnL": round(p.unrealized_pnl, 2)}
            for s, p in self.broker.positions().items() if p.quantity != 0
        }
        locked = bool(day["locked"])
        return {
            "mode": self.broker.name,
            "live_data": self.md.using_live_data,
            "realized_pnl": round(float(day["realized_pnl"]), 2),
            "unrealized_pnl": round(self.risk.unrealized_pnl(), 2),
            "total_pnl": round(self.risk.total_pnl(), 2),
            "trade_count": int(day["trade_count"]),
            "locked": locked,
            "lock_reason": day["lock_reason"],
            "status": "LOCKED" if locked else "ACTIVE",
            "profit_target": self.risk.profit_target,
            "max_loss": self.risk.max_loss,
            "positions": positions,
        }



