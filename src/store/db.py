"""SQLite persistence: orders, fills, per-day state (P&L + lock), audit log.

The daily_state row is the source of truth for the "locked for the day"
kill-switch flag, so a restart never silently resumes trading after a limit.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

from ..config import DATA_DIR

DB_PATH = DATA_DIR / "tradingbot.db"


SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    trade_date TEXT NOT NULL,
    strategy TEXT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,            -- BUY / SELL
    quantity INTEGER NOT NULL,
    order_type TEXT NOT NULL,      -- MARKET / LIMIT
    price REAL,                    -- requested (limit) price
    fill_price REAL,               -- executed price
    status TEXT NOT NULL,          -- FILLED / REJECTED / PENDING
    broker_order_id TEXT,
    mode TEXT NOT NULL,            -- paper / live
    reason TEXT
);

CREATE TABLE IF NOT EXISTS daily_state (
    trade_date TEXT PRIMARY KEY,
    realized_pnl REAL NOT NULL DEFAULT 0,
    trade_count INTEGER NOT NULL DEFAULT 0,
    locked INTEGER NOT NULL DEFAULT 0,     -- 1 = no more trading today
    lock_reason TEXT,
    started INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    event TEXT NOT NULL,
    detail TEXT
);
"""


class Store:
    def __init__(self, path: Path = DB_PATH):
        self.path = path
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ----- orders -----
    def record_order(self, **kw: Any) -> int:
        cols = (
            "ts", "trade_date", "strategy", "symbol", "side", "quantity",
            "order_type", "price", "fill_price", "status", "broker_order_id",
            "mode", "reason",
        )
        kw.setdefault("ts", datetime.now().isoformat(timespec="seconds"))
        kw.setdefault("trade_date", date.today().isoformat())
        values = [kw.get(c) for c in cols]
        placeholders = ", ".join("?" for _ in cols)
        cur = self.conn.execute(
            f"INSERT INTO orders ({', '.join(cols)}) VALUES ({placeholders})", values
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def orders_today(self) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT * FROM orders WHERE trade_date = ? ORDER BY id",
                (date.today().isoformat(),),
            )
        )

    # ----- daily state -----
    def _ensure_day(self, d: Optional[str] = None) -> str:
        d = d or date.today().isoformat()
        self.conn.execute(
            "INSERT OR IGNORE INTO daily_state (trade_date, updated_at) VALUES (?, ?)",
            (d, datetime.now().isoformat(timespec="seconds")),
        )
        self.conn.commit()
        return d

    def get_day(self, d: Optional[str] = None) -> sqlite3.Row:
        d = self._ensure_day(d)
        return self.conn.execute(
            "SELECT * FROM daily_state WHERE trade_date = ?", (d,)
        ).fetchone()

    def update_day(self, d: Optional[str] = None, **fields: Any) -> None:
        d = self._ensure_day(d)
        fields["updated_at"] = datetime.now().isoformat(timespec="seconds")
        sets = ", ".join(f"{k} = ?" for k in fields)
        self.conn.execute(
            f"UPDATE daily_state SET {sets} WHERE trade_date = ?",
            [*fields.values(), d],
        )
        self.conn.commit()

    def add_realized(self, amount: float, d: Optional[str] = None) -> None:
        row = self.get_day(d)
        self.update_day(d, realized_pnl=row["realized_pnl"] + amount)

    def incr_trades(self, d: Optional[str] = None) -> None:
        row = self.get_day(d)
        self.update_day(d, trade_count=row["trade_count"] + 1)

    def lock_day(self, reason: str, d: Optional[str] = None) -> None:
        self.update_day(d, locked=1, lock_reason=reason)
        self.audit("DAY_LOCKED", reason)

    def unlock_day(self, d: Optional[str] = None) -> None:
        self.update_day(d, locked=0, lock_reason=None)
        self.audit("DAY_UNLOCKED", d or date.today().isoformat())

    def is_locked(self, d: Optional[str] = None) -> bool:
        return bool(self.get_day(d)["locked"])

    # ----- audit -----
    def audit(self, event: str, detail: Any = None) -> None:
        if not isinstance(detail, str):
            detail = json.dumps(detail, default=str)
        self.conn.execute(
            "INSERT INTO audit (ts, event, detail) VALUES (?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), event, detail),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
