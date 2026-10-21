"""Indicators.

Two families:
  * list[float] helpers (sma/ema/...) used by the older tick-series strategies.
  * DataFrame helpers (vwap/rsi/atr/supertrend) used by the OHLCV bar
    strategies (orb_vwap / supertrend_vwap / rsi_reversion). These operate on
    a pandas frame with columns: date, open, high, low, close, volume.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(prices: list[float], period: int) -> float | None:
    if len(prices) < period:
        return None
    return float(np.mean(prices[-period:]))


def ema(prices: list[float], period: int) -> float | None:
    if len(prices) < period:
        return None
    arr = np.asarray(prices[-(period * 3):], dtype=float)
    k = 2 / (period + 1)
    e = arr[0]
    for p in arr[1:]:
        e = p * k + e * (1 - k)
    return float(e)


def rolling_std(prices: list[float], period: int) -> float | None:
    if len(prices) < period:
        return None
    return float(np.std(prices[-period:]))


def supertrend_lite(prices: list[float], period: int = 10, mult: float = 3.0):
    """A tick-series approximation of Supertrend.

    Real Supertrend uses ATR from OHLC candles; on a pure tick/close series
    we use rolling volatility as the band width. Returns (trend, lower, upper)
    where trend is +1 (up) / -1 (down) / 0 (not enough data).
    """
    if len(prices) < period + 1:
        return 0, None, None
    mid = ema(prices, period)
    vol = rolling_std(prices, period)
    if mid is None or vol is None:
        return 0, None, None
    upper = mid + mult * vol
    lower = mid - mult * vol
    last = prices[-1]
    trend = 1 if last >= mid else -1
    return trend, lower, upper


# =====================================================================
#  DataFrame (OHLCV) indicators
# =====================================================================

def _session_key(df: pd.DataFrame) -> pd.Series:
    """Group bars by calendar day so VWAP / opening-range reset each session."""
    if "date" in df.columns:
        return pd.to_datetime(df["date"]).dt.date
    return pd.Series([0] * len(df), index=df.index)


def vwap(df: pd.DataFrame) -> pd.Series:
    """Session-anchored VWAP. Resets each trading day. If volume is missing
    or zero (e.g. live ticks without a data subscription), gracefully falls
    back to a running mean of the typical price."""
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df.get("volume")
    if vol is None:
        vol = pd.Series(np.zeros(len(df)), index=df.index)
    vol = vol.fillna(0.0)
    keys = _session_key(df)
    out = pd.Series(index=df.index, dtype=float)
    for _, idx in df.groupby(keys).groups.items():
        t = typical.loc[idx]
        v = vol.loc[idx]
        cum_v = v.cumsum()
        cum_tv = (t * v).cumsum()
        # Where cumulative volume is 0, use expanding mean of typical price.
        vw = cum_tv / cum_v.replace(0, np.nan)
        vw = vw.fillna(t.expanding().mean())
        out.loc[idx] = vw
    return out


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.Series:
    """MACD line = EMA(fast) - EMA(slow) of close. Positive = bullish momentum."""
    close = df["close"]
    return (close.ewm(span=fast, adjust=False).mean()
            - close.ewm(span=slow, adjust=False).mean())


def supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> pd.Series:
    """Real Supertrend from OHLC + ATR. Returns a trend series: +1 up / -1 down."""
    hl2 = (df["high"] + df["low"]) / 2.0
    a = atr(df, period)
    upper = hl2 + mult * a
    lower = hl2 - mult * a
    close = df["close"].to_numpy()
    up = upper.to_numpy()
    lo = lower.to_numpy()
    n = len(df)
    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    trend = np.ones(n, dtype=int)
    for i in range(n):
        if i == 0 or np.isnan(up[i]) or np.isnan(lo[i]):
            final_upper[i] = up[i]
            final_lower[i] = lo[i]
            trend[i] = 1
            continue
        final_upper[i] = (up[i] if (up[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1])
                          else final_upper[i - 1])
        final_lower[i] = (lo[i] if (lo[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1])
                          else final_lower[i - 1])
        if close[i] > final_upper[i - 1]:
            trend[i] = 1
        elif close[i] < final_lower[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]
    return pd.Series(trend, index=df.index)
