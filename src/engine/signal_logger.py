"""Paper-live signal logger — the honest way to evaluate a live-data strategy.

Runs the bot in PAPER mode (simulated fills) but with LIVE logic and data:
real (delayed) prices, real fundamentals, real news sentiment, real VIX. It
writes two CSVs you can open in Excel to judge the strategy over real days:

  logs/signals_YYYY-MM-DD.csv  — every order the bot placed (with its reason)
  logs/equity_YYYY-MM-DD.csv   — a P&L / positions snapshot each poll

No real money is ever at risk. This answers the one question a fundamentals
strategy (HAASEY v3) can't answer in backtest: does it work on live data?

Run during market hours (IST) for meaningful movement; outside hours the
prices are static (last close) so you'll mostly see snapshots, not trades.
"""
from __future__ import annotations

import csv
import time
from datetime import date, datetime

from ..config import LOG_DIR, Config
from .trader import Trader, _parse_hhmm


class PaperLiveLogger:
    def __init__(self, config: Config, use_live_quotes: bool = True):
        self.config = config
        self.trader = Trader(config)
        if use_live_quotes:
            self.trader.md.use_live_quotes = True  # real-ish prices instead of synthetic
        day = date.today().isoformat()
        self.sig_path = LOG_DIR / f"signals_{day}.csv"
        self.eq_path = LOG_DIR / f"equity_{day}.csv"
        self._last_order_id = 0

    def _init_csv(self) -> None:
        if not self.sig_path.exists():
            with open(self.sig_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    ["ts", "strategy", "symbol", "side", "qty", "order_type",
                     "fill_price", "status", "mode", "reason"])
        if not self.eq_path.exists():
            with open(self.eq_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    ["ts", "realized_pnl", "unrealized_pnl", "total_pnl",
                     "open_positions", "trade_count", "locked", "lock_reason"])

    def _export_new_orders(self) -> int:
        rows = self.trader.store.orders_today()
        new = [r for r in rows if r["id"] > self._last_order_id]
        if new:
            with open(self.sig_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                for r in new:
                    w.writerow([r["ts"], r["strategy"], r["symbol"], r["side"],
                                r["quantity"], r["order_type"], r["fill_price"],
                                r["status"], r["mode"], r["reason"]])
            self._last_order_id = max(r["id"] for r in new)
        return len(new)

    def _snapshot(self) -> dict:
        s = self.trader.status()
        with open(self.eq_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                datetime.now().isoformat(timespec="seconds"),
                s["realized_pnl"], s["unrealized_pnl"], s["total_pnl"],
                len(s["positions"]), s["trade_count"], int(s["locked"]),
                s["lock_reason"] or ""])
        return s

    def run(self, minutes: float | None = None, poll_seconds: float = 60.0,
            respect_session: bool = True) -> None:
        self.trader.connect()
        self._init_csv()
        sess = self.config.session
        t_start = _parse_hhmm(sess.get("start_trading", "09:20"))
        t_stop_new = _parse_hhmm(sess.get("stop_new_trades", "15:10"))
        t_sq = _parse_hhmm(sess.get("square_off_all", "15:20"))
        started = time.time()
        live_data = "yfinance live/delayed" if self.trader.md.use_live_quotes else "synthetic"
        print(f"[paperlog] LIVE-PAPER mode | prices={live_data} | poll={poll_seconds}s")
        print(f"[paperlog] signals -> {self.sig_path}")
        print(f"[paperlog] equity  -> {self.eq_path}")

        while True:
            if minutes and (time.time() - started) >= minutes * 60:
                print("[paperlog] duration reached — stopping.")
                break
            if respect_session:
                now_t = self.trader.now().time()
                if now_t >= t_sq:
                    self.trader.square_off_eod("paperlog EOD square-off")
                    self._export_new_orders()
                    self._snapshot()
                    print("[paperlog] market close — squared off, stopping.")
                    break
                if now_t < t_start:
                    time.sleep(poll_seconds)
                    continue
                allow_new = now_t < t_stop_new
            else:
                allow_new = True

            self.trader.run_once(allow_new=allow_new)
            n = self._export_new_orders()
            s = self._snapshot()
            if n:
                print(f"[paperlog] {datetime.now():%H:%M:%S}  +{n} order(s)  "
                      f"P&L Rs.{s['total_pnl']:,.0f}  positions {len(s['positions'])}")
            if self.trader.risk.is_locked():
                print("[paperlog] daily limit hit — day locked, stopping.")
                break
            time.sleep(poll_seconds)
        print("[paperlog] done. Open the CSVs to review signals and the P&L curve.")
