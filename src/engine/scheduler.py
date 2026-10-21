"""Daily automation: auto-start the session each trading morning and force a
square-off near the close. Use this for the "set it and forget it" daemon.

  python -m src.main daemon

Note: it auto-starts each NEW day, but if a daily limit locked the previous
session, that lock is per-day — a fresh day starts clean. Within a day, a
limit lock stays until you manually reset (your requirement #1).
"""
from __future__ import annotations

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from ..config import Config
from .trader import Trader


def _hhmm(s: str) -> tuple[int, int]:
    h, m = s.split(":")
    return int(h), int(m)


def run_daemon(config: Config, poll_seconds: float = 2.0) -> None:
    sess = config.session
    tz = sess.get("timezone", "Asia/Kolkata")
    sh, sm = _hhmm(sess.get("start_trading", "09:20"))
    eh, em = _hhmm(sess.get("square_off_all", "15:20"))
    dow = "mon-fri" if not sess.get("trade_on_weekends") else "*"

    scheduler = BackgroundScheduler(timezone=tz)

    def start_session():
        print("[daemon] market open — starting trading session")
        Trader(config).run_session(poll_seconds=poll_seconds, respect_session=True)

    def force_squareoff():
        print("[daemon] forcing end-of-day square-off")
        t = Trader(config)
        t.connect()
        t.square_off_eod("daemon EOD square-off")

    scheduler.add_job(start_session, CronTrigger(day_of_week=dow, hour=sh, minute=sm))
    scheduler.add_job(force_squareoff, CronTrigger(day_of_week=dow, hour=eh, minute=em))
    scheduler.start()
    print(f"[daemon] scheduled: start {sh:02d}:{sm:02d}, square-off {eh:02d}:{em:02d} ({dow}, {tz})")
    print("[daemon] running. Ctrl+C to stop.")
    try:
        import time as _t
        while True:
            _t.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        print("[daemon] stopped.")
