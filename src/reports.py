"""Read a paper-live equity/signals CSV and print honest performance stats.

Deliberately does NOT annualise a single intraday session into a headline
Sharpe — that would be misleading. It reports what the data actually supports:
net P&L, peak/trough, max drawdown, and (only across multiple days) a daily
Sharpe.
"""
from __future__ import annotations

import csv
import glob
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

from .config import LOG_DIR


def _latest_equity_csv() -> str | None:
    files = sorted(glob.glob(str(LOG_DIR / "equity_*.csv")))
    return files[-1] if files else None


def analyze_equity(path: str | None = None, capital: float = 20000.0) -> dict | None:
    path = path or _latest_equity_csv()
    if not path or not Path(path).exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    pnl = [float(r["total_pnl"]) for r in rows]
    ts = [r["ts"] for r in rows]
    final = pnl[-1]
    peak_val, trough_val = max(pnl), min(pnl)

    running_peak = pnl[0]
    max_dd = 0.0
    for p in pnl:
        running_peak = max(running_peak, p)
        max_dd = min(max_dd, p - running_peak)

    # Per-day P&L for a daily Sharpe (only meaningful with >=3 days).
    by_day: dict[str, float] = {}
    for r in rows:
        d = r["ts"][:10]
        by_day[d] = float(r["total_pnl"])
    day_end = list(by_day.values())
    daily_sharpe = None
    # Require a real sample (>=15 days). Fewer days gives absurd Sharpes
    # (tiny-variance artifacts like 11.x) that mislead — so we don't show them.
    if len(day_end) >= 15:
        rets = [(day_end[i] - day_end[i - 1]) / capital for i in range(1, len(day_end))]
        sd = pstdev(rets)
        if sd > 0:
            daily_sharpe = (mean(rets) / sd) * (252 ** 0.5)

    trade_count = int(rows[-1].get("trade_count", 0) or 0)
    return {
        "path": path,
        "snapshots": len(rows),
        "days": len(by_day),
        "from": ts[0],
        "to": ts[-1],
        "net_pnl": round(final, 2),
        "return_pct": round(100 * final / capital, 3),
        "peak": round(peak_val, 2),
        "trough": round(trough_val, 2),
        "max_drawdown": round(max_dd, 2),
        "max_drawdown_pct": round(100 * max_dd / capital, 3),
        "trade_count": trade_count,
        "daily_sharpe": None if daily_sharpe is None else round(daily_sharpe, 2),
    }


def _signals_summary(equity_path: str) -> dict:
    sig_path = equity_path.replace("equity_", "signals_")
    if not Path(sig_path).exists():
        return {}
    with open(sig_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    buys = sum(1 for r in rows if r["side"] == "BUY")
    sells = sum(1 for r in rows if r["side"] == "SELL")
    syms = sorted({r["symbol"] for r in rows})
    return {"orders": len(rows), "buys": buys, "sells": sells, "symbols": syms}


def format_report(stats: dict) -> str:
    sig = _signals_summary(stats["path"])
    lines = [
        "",
        f"===== BOT PERFORMANCE — {Path(stats['path']).name} =====",
        f"  Window        : {stats['from']}  ->  {stats['to']}  ({stats['days']} day(s), {stats['snapshots']} snapshots)",
        f"  Net P&L       : Rs.{stats['net_pnl']:,.2f}   ({stats['return_pct']:+.3f}% of capital)",
        f"  Peak / Trough : Rs.{stats['peak']:,.2f} / Rs.{stats['trough']:,.2f}",
        f"  Max Drawdown  : Rs.{stats['max_drawdown']:,.2f}  ({stats['max_drawdown_pct']:.3f}%)",
        f"  Trades        : {stats['trade_count']}",
    ]
    if sig:
        lines.append(f"  Orders logged : {sig['orders']} ({sig['buys']} buy / {sig['sells']} sell) "
                     f"on {', '.join(sig['symbols'])}")
    if stats["daily_sharpe"] is not None:
        lines.append(f"  Daily Sharpe  : {stats['daily_sharpe']} (annualised)")
    else:
        lines.append(f"  Daily Sharpe  : n/a — only {stats['days']} day(s); need ~15+ for a "
                     "non-misleading figure")
    lines.append("  NOTE: paper-live on ~15-min-delayed free data. A few sessions is a small")
    lines.append("        sample — judge the strategy only after multiple full trading days.")
    return "\n".join(lines)
