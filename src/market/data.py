"""Market data: live LTP + historical candles.

Source priority:
  1. Zerodha Kite (real quotes) — used when a Kite session is available.
  2. Synthetic random-walk — so PAPER mode and backtests run anywhere,
     even with no data subscription / outside market hours. Synthetic
     prices are clearly flagged and must never be used for live trading.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd


@dataclass
class _SynthState:
    price: float
    drift: float = 0.0
    vol: float = 0.004  # ~0.4% per step


class MarketData:
    def __init__(self, config):
        self.config = config
        self._kite = None
        self._instrument_cache: dict[str, int] = {}
        self._synth: dict[str, _SynthState] = {}
        self.exchange = config.account.get("exchange", "NSE")
        self.last_source = "synthetic"  # set by historical(): kite / yfinance / synthetic
        # Free live/delayed quotes via yfinance (opt-in, e.g. for paper-live logging).
        self.use_live_quotes = False
        self._live_cache: dict[str, tuple[float, float]] = {}
        self._live_ttl = 45.0
        # NSE index symbols -> Yahoo Finance tickers (equities just get ".NS").
        self._yahoo_index = {
            "NIFTY 50": "^NSEI", "NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK",
            "NIFTY BANK": "^NSEBANK", "SENSEX": "^BSESN",
        }
        # Reasonable seed prices so synthetic mode looks realistic.
        self._seed = {
            "GAIL": 150.0, "RELIANCE": 2900.0, "TCS": 3900.0, "INFY": 1850.0,
            "SBIN": 820.0, "HDFCBANK": 1650.0, "NIFTY 50": 24000.0,
        }

    # ----- connection -----
    def connect(self) -> None:
        s = self.config.secrets
        if s.has_kite:
            try:
                from kiteconnect import KiteConnect

                self._kite = KiteConnect(api_key=s.kite_api_key)
                self._kite.set_access_token(s.kite_access_token)
            except Exception:
                self._kite = None

    @property
    def using_live_data(self) -> bool:
        return self._kite is not None

    def _instrument_token(self, symbol: str) -> Optional[int]:
        if symbol in self._instrument_cache:
            return self._instrument_cache[symbol]
        if not self._kite:
            return None
        try:
            data = self._kite.ltp([f"{self.exchange}:{symbol}"])
            token = data[f"{self.exchange}:{symbol}"]["instrument_token"]
            self._instrument_cache[symbol] = token
            return token
        except Exception:
            return None

    # ----- live price -----
    def ltp(self, symbol: str) -> float:
        if self._kite:
            try:
                key = f"{self.exchange}:{symbol}"
                return float(self._kite.ltp([key])[key]["last_price"])
            except Exception:
                pass
        if self.use_live_quotes:
            v = self._live_quote(symbol)
            if v is not None:
                return v
        return self._synthetic_tick(symbol)

    def _live_quote(self, symbol: str) -> Optional[float]:
        """Free (delayed) last price from yfinance, cached briefly. Real during
        market hours; returns the last close when the market is shut."""
        hit = self._live_cache.get(symbol)
        if hit and (time.time() - hit[0] < self._live_ttl):
            return hit[1]
        price = None
        try:
            import yfinance as yf

            fi = yf.Ticker(self._yahoo_ticker(symbol)).fast_info
            price = getattr(fi, "last_price", None)
            if price is None:
                try:
                    price = fi["lastPrice"]
                except Exception:
                    price = None
            price = float(price) if price else None
        except Exception:
            price = None
        if price:
            self._live_cache[symbol] = (time.time(), price)
        return price

    def _synthetic_tick(self, symbol: str) -> float:
        st = self._synth.get(symbol)
        if st is None:
            st = _SynthState(price=self._seed.get(symbol, 100.0))
            self._synth[symbol] = st
        # Geometric random walk with mild mean reversion.
        shock = random.gauss(st.drift, st.vol)
        st.price = max(0.5, st.price * (1 + shock))
        return round(st.price, 2)

    # ----- historical candles (for strategies / backtests) -----
    def historical(self, symbol: str, interval: str = "minute", days: int = 1) -> pd.DataFrame:
        # 1) Kite (paid data subscription) — best, official.
        if self._kite:
            token = self._instrument_token(symbol)
            if token:
                try:
                    to = datetime.now()
                    frm = to - timedelta(days=days)
                    candles = self._kite.historical_data(token, frm, to, interval)
                    df = pd.DataFrame(candles)
                    if not df.empty:
                        self.last_source = "kite"
                        return df
                except Exception:
                    pass
        # 2) Yahoo Finance via yfinance — FREE real NSE history.
        df = self._yahoo_candles(symbol, days=days)
        if df is not None and not df.empty:
            self.last_source = "yfinance"
            return df
        # 3) Synthetic random-walk — plumbing only, never a profit claim.
        self.last_source = "synthetic"
        return self._synthetic_candles(symbol, bars=days * 375)  # ~375 min/day

    def _yahoo_ticker(self, symbol: str) -> str:
        s = symbol.upper().strip()
        return self._yahoo_index.get(s, f"{s}.NS")

    def _yahoo_candles(self, symbol: str, days: int = 5) -> Optional[pd.DataFrame]:
        try:
            import yfinance as yf
        except Exception:
            return None
        ticker = self._yahoo_ticker(symbol)
        # yfinance: 1m data only for the last ~7 days; use 5m for longer windows.
        interval = "1m" if days <= 7 else "5m"
        period = f"{min(days, 7)}d" if interval == "1m" else f"{min(days, 60)}d"
        try:
            raw = yf.download(ticker, period=period, interval=interval,
                              progress=False, auto_adjust=False)
        except Exception:
            return None
        if raw is None or raw.empty:
            return None
        raw = raw.reset_index()
        # yfinance may return MultiIndex columns for a single ticker — flatten.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] for c in raw.columns]
        rename = {"Datetime": "date", "Date": "date", "Open": "open", "High": "high",
                  "Low": "low", "Close": "close", "Volume": "volume"}
        raw = raw.rename(columns=rename)
        cols = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in raw.columns]
        return raw[cols].dropna(subset=["close"])

    def _synthetic_candles(self, symbol: str, bars: int = 375) -> pd.DataFrame:
        price = self._seed.get(symbol, 100.0)
        rows = []
        t = datetime.now() - timedelta(minutes=bars)
        for _ in range(bars):
            o = price
            shock = random.gauss(0, 0.004)
            c = max(0.5, o * (1 + shock))
            hi = max(o, c) * (1 + abs(random.gauss(0, 0.002)))
            lo = min(o, c) * (1 - abs(random.gauss(0, 0.002)))
            rows.append({"date": t, "open": round(o, 2), "high": round(hi, 2),
                         "low": round(lo, 2), "close": round(c, 2),
                         "volume": random.randint(1000, 50000)})
            price = c
            t += timedelta(minutes=1)
        return pd.DataFrame(rows)
