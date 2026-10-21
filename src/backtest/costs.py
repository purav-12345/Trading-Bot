"""Transaction cost + slippage model for Indian intraday equity (Zerodha MIS).

Why this matters: a backtest that ignores costs is a lie. A strategy that
churns hundreds of trades can look flat on raw prices and bleed badly once
brokerage + STT + exchange + GST + stamp + slippage are subtracted. This
model makes paper P&L realistic.

Rates are the Zerodha intraday-equity defaults as of 2025-2026 and are
configurable in settings.yaml -> costs (they change; verify against
https://zerodha.com/charges before trusting live numbers).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostModel:
    # Brokerage: 0.03% of turnover or flat cap, whichever is lower, PER order.
    brokerage_pct: float = 0.0003
    brokerage_cap: float = 20.0
    # STT: 0.025% on the SELL side only (intraday equity).
    stt_pct_sell: float = 0.00025
    # NSE exchange transaction charge on turnover (both sides).
    exchange_txn_pct: float = 0.0000297
    # SEBI charges on turnover (both sides): Rs.10 per crore.
    sebi_pct: float = 0.000001
    # Stamp duty on the BUY side only.
    stamp_pct_buy: float = 0.00003
    # GST: 18% on (brokerage + exchange txn + sebi).
    gst_pct: float = 0.18
    # Slippage assumption (basis points of price) applied adversely per fill.
    slippage_bps: float = 2.0

    # ------------------------------------------------------------------
    def fill_with_slippage(self, price: float, side: str) -> float:
        """Buys fill a touch higher, sells a touch lower — the realistic enemy."""
        slip = price * (self.slippage_bps / 10000.0)
        return price + slip if side.upper() == "BUY" else price - slip

    def charges(self, price: float, qty: int, side: str) -> float:
        """Total statutory + broker charges for ONE executed order."""
        turnover = price * qty
        brokerage = min(self.brokerage_pct * turnover, self.brokerage_cap)
        stt = self.stt_pct_sell * turnover if side.upper() == "SELL" else 0.0
        exch = self.exchange_txn_pct * turnover
        sebi = self.sebi_pct * turnover
        stamp = self.stamp_pct_buy * turnover if side.upper() == "BUY" else 0.0
        gst = self.gst_pct * (brokerage + exch + sebi)
        return brokerage + stt + exch + sebi + stamp + gst

    @classmethod
    def from_config(cls, config) -> "CostModel":
        c = (config.settings.get("costs") or {})
        return cls(**{k: v for k, v in c.items() if k in cls.__dataclass_fields__})
