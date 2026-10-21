"""Multi-factor scoring engine for the HAASEY strategy.

Combines sub-scores (each normalised to roughly -1..+1) into a weighted total:

    total = technical*wt + fundamental*wf + sentiment*ws
                + momentum*wm + macro*wx

Missing components (e.g. fundamentals unavailable, or backtest mode where
point-in-time-now data would be look-ahead bias) are dropped and the weights
are renormalised over the components actually used — so an absent factor
doesn't silently bias the score toward zero.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


# ---------------------------------------------------------------------------
# Sub-scores
# ---------------------------------------------------------------------------
def technical_score(df: pd.DataFrame) -> Optional[float]:
    """Blend of trend (EMA stack), RSI, MACD, Bollinger position."""
    if len(df) < 60:
        return None
    close = df["close"]
    price = float(close.iloc[-1])
    ema20 = float(_ema(close, 20).iloc[-1])
    ema50 = float(_ema(close, 50).iloc[-1])

    trend = 1.0 if price > ema20 > ema50 else (-1.0 if price < ema20 < ema50 else 0.0)

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = float((100 - 100 / (1 + rs)).fillna(50).iloc[-1])
    rsi_s = _clip((rsi - 50) / 40)  # +1 near 90, -1 near 10

    # MACD sign (12/26/9)
    macd = _ema(close, 12) - _ema(close, 26)
    signal = _ema(macd, 9)
    macd_s = _clip(float((macd - signal).iloc[-1]) / (abs(float(macd.iloc[-1])) + 1e-6))

    # Bollinger position (20, 2sd) -> -1 at lower band, +1 at upper
    ma20 = close.rolling(20).mean()
    sd20 = close.rolling(20).std()
    lower = float((ma20 - 2 * sd20).iloc[-1])
    upper = float((ma20 + 2 * sd20).iloc[-1])
    boll_s = _clip((price - (upper + lower) / 2) / ((upper - lower) / 2 + 1e-6))

    return _clip((trend + rsi_s + macd_s + boll_s) / 4)


def momentum_score(df: pd.DataFrame, lookback: int = 20) -> Optional[float]:
    if len(df) < lookback + 1:
        return None
    close = df["close"]
    ret = float(close.iloc[-1] / close.iloc[-lookback - 1] - 1.0)
    return _clip(math.tanh(ret * 20))  # ~+/-2.5% saturates


def fundamental_score(f: Optional[dict]) -> Optional[float]:
    """Heuristic from common ratios. Higher ROE / margins / lower debt = better."""
    if not f:
        return None
    parts = []
    pe = f.get("trailingPE")
    if pe and pe > 0:
        parts.append(_clip((30 - pe) / 30))          # cheaper PE scores higher
    roe = f.get("returnOnEquity")
    if roe is not None:
        parts.append(_clip(roe / 0.25))              # 25% ROE -> +1
    de = f.get("debtToEquity")
    if de is not None:
        parts.append(_clip((100 - de) / 100))        # lower leverage better
    margin = f.get("profitMargins")
    if margin is not None:
        parts.append(_clip(margin / 0.20))           # 20% margin -> +1
    if not parts:
        return None
    return _clip(sum(parts) / len(parts))


def macro_score(vix: Optional[float]) -> Optional[float]:
    """Risk-on/off from India VIX. Low VIX (calm) positive, high VIX negative."""
    if vix is None:
        return None
    return _clip((18.0 - vix) / 10.0)  # ~+1 at VIX 8, -1 at VIX 28


@dataclass
class ScoreResult:
    total: float
    breakdown: dict = field(default_factory=dict)

    def signal(self, strong=0.70, buy=0.55) -> str:
        if self.total >= strong:
            return "STRONG_BUY"
        if self.total >= buy:
            return "BUY"
        if self.total <= -strong:
            return "STRONG_SELL"
        if self.total <= -buy:
            return "SELL"
        return "HOLD"


class MultiFactorScorer:
    def __init__(self, weights: Optional[dict] = None):
        self.weights = weights or {
            "technical": 0.30, "fundamental": 0.25, "sentiment": 0.20,
            "momentum": 0.15, "macro": 0.10,
        }

    def score(self, df: pd.DataFrame, fundamentals=None, sentiment=None,
              vix=None) -> ScoreResult:
        comps = {
            "technical": technical_score(df),
            "fundamental": fundamental_score(fundamentals),
            "sentiment": None if sentiment is None else _clip(float(sentiment)),
            "momentum": momentum_score(df),
            "macro": macro_score(vix),
        }
        used = {k: v for k, v in comps.items() if v is not None}
        if not used:
            return ScoreResult(0.0, comps)
        wsum = sum(self.weights.get(k, 0) for k in used) or 1.0
        total = sum(self.weights.get(k, 0) * v for k, v in used.items()) / wsum
        return ScoreResult(_clip(total), comps)
