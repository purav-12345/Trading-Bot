"""HAASEY v2 — a fully-autonomous multi-factor stock picker (user-designed).

v2 fixes v1's over-trading (417 trades / 5 days, where costs > price loss).
Six changes:
  1. Entry-only scoring. Exits are driven by an ATR TRAILING STOP, not by the
     score flipping — so once in, it rides the trend until it breaks.
  2. Minimum holding period (cooldown): cannot exit within `cooldown_minutes`.
  3. Trend filter: only buy if price > SMA(trend_sma) AND price > EMA(trend_ema).
  4. Stricter entries: BUY >= 0.55, STRONG_BUY >= 0.70 (0.30-0.55 = HOLD).
  5. Decisions on `decision_bar_minutes` (default 15-min) resampled candles,
     acted on once per newly-COMPLETED bar (kills 5-min noise churn).
  6. Cost-aware sizing: skip a trade unless expected move (target_atr_mult x
     ATR) beats the estimated round-trip cost by `min_profit_cost_ratio`.

Honesty guards unchanged: backtest uses technical+momentum only (no
look-ahead); long-only equity; options are not auto-traded. The risk engine
and net-worth kill switch still govern everything.
"""
from __future__ import annotations

import pandas as pd

from . import indicators as ind
from .base import Signal, Strategy, StrategyContext
from .scoring import MultiFactorScorer


def position_size(capital: float, entry: float, atr: float, strength: float,
                  per_trade_risk_pct: float, max_pos_pct: float) -> int:
    if entry <= 0:
        return 0
    risk_per_share = max(1.5 * atr, entry * 0.002)
    risk_amount = capital * per_trade_risk_pct * (0.5 + 0.5 * abs(strength))
    qty = int(risk_amount / risk_per_share)
    max_qty = int((capital * max_pos_pct) / entry)
    return max(0, min(qty, max_qty))


def _resample(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Aggregate raw bars into `minutes`-minute OHLCV decision candles."""
    d = df.copy()
    d["ts"] = pd.to_datetime(d["date"])
    d = d.set_index("ts")
    agg = d.resample(f"{minutes}min").agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last", "volume": "sum"}).dropna()
    return agg.reset_index().rename(columns={"ts": "date"})


def _round_trip_cost(price: float, qty: int) -> float:
    """Rough Zerodha intraday round-trip cost (buy+sell) for the sizing gate."""
    turnover = price * qty
    brokerage = min(0.0003 * turnover, 20.0) * 2       # both legs, Rs.20 cap each
    stt = 0.00025 * turnover                            # sell side
    exch_gst = 0.0000297 * turnover * 2 * 1.18
    slippage = (2.0 / 10000.0) * turnover * 2           # 2 bps each side
    return brokerage + stt + exch_gst + slippage


class HaaseyStrategy(Strategy):
    type = "haasey"

    def __init__(self, cfg: dict):
        uni = cfg.get("universe")
        if not uni or uni == "auto":
            from ..data import default_universe
            uni = default_universe()
        cfg.setdefault("symbol", uni[0])
        super().__init__(cfg)
        self.universe = uni
        self.capital = float(cfg.get("capital", 20000))
        self.max_positions = int(cfg.get("max_positions", 5))
        self.per_trade_risk_pct = float(cfg.get("per_trade_risk_pct", 0.02))
        self.max_pos_pct = float(cfg.get("max_pos_pct", 0.20))
        self.atr_period = int(cfg.get("atr_period", 14))
        # v2 params
        self.decision_minutes = int(cfg.get("decision_bar_minutes", 15))
        self.cooldown_minutes = int(cfg.get("cooldown_minutes", 60))
        self.trail_atr_mult = float(cfg.get("trail_atr_mult", 2.0))
        self.target_atr_mult = float(cfg.get("target_atr_mult", 3.0))
        self.min_profit_cost_ratio = float(cfg.get("min_profit_cost_ratio", 1.6))
        self.trend_sma = int(cfg.get("trend_sma", 50))
        self.trend_ema = int(cfg.get("trend_ema", 20))
        # stricter defaults
        self.strong_buy = float(cfg.get("strong_buy", 0.70))
        self.buy = float(cfg.get("buy", 0.55))
        self.use_fundamentals = bool(cfg.get("use_fundamentals", True))
        self.use_sentiment = bool(cfg.get("use_sentiment", True))
        self.use_macro = bool(cfg.get("use_macro", True))
        # Market-regime filter: skip ALL trades when the index is weak.
        self.use_regime = bool(cfg.get("use_regime", True))
        self.regime_symbol = cfg.get("regime_symbol", "NIFTY 50")
        self.regime_sma = int(cfg.get("regime_sma", 20))
        self.regime_tol = float(cfg.get("regime_tol", 0.98))
        self.scorer = MultiFactorScorer(cfg.get("weights"))
        self._fund = None
        self._prebucketed = False  # set by the backtester (bars already resampled)
        # per-symbol runtime state
        self._state: dict[str, dict] = {}      # sym -> {entry_ts, peak, atr}
        self._last_bar: dict[str, object] = {}  # sym -> last completed decision-bar ts

    def set_prebucketed(self, flag: bool = True) -> None:
        self._prebucketed = flag

    def _decision_df(self, raw):
        """Decision candles. Live: resample raw to decision_minutes and drop the
        still-forming last bar. Backtest (prebucketed): raw already = completed
        decision bars, so use as-is (huge speedup, no per-tick resampling)."""
        if self._prebucketed:
            return raw
        rdf = _resample(raw, self.decision_minutes)
        return rdf.iloc[:-1] if len(rdf) > 1 else rdf

    def _fundamentals(self, symbol):
        if self._fund is None:
            from ..data import FundamentalsProvider
            self._fund = FundamentalsProvider()
        return self._fund.get(symbol)

    # ------------------------------------------------------------------
    def on_tick(self, ctx: StrategyContext) -> list[Signal]:
        live = not ctx.backtest
        vix = None
        if live and self.use_macro:
            try:
                from ..data import india_vix
                vix = india_vix()
            except Exception:
                vix = None

        # Improvement 3b: market-regime filter. If the index is >regime_tol
        # below its SMA, the whole market is weak -> take NO new trades.
        # (Existing positions are still managed/exited below.)
        market_weak = self._market_weak(ctx)

        signals: list[Signal] = []
        candidates: list[tuple] = []
        deployed = 0.0
        open_count = 0

        for sym in self.universe:
            raw = ctx.bars(sym)
            if len(raw) < 60:
                continue
            completed = self._decision_df(raw)
            if len(completed) < max(self.trend_sma, 60) + 1:
                continue
            last_ts = completed["date"].iloc[-1]
            new_bar = self._last_bar.get(sym) != last_ts
            self._last_bar[sym] = last_ts

            price = float(completed["close"].iloc[-1])
            pos = ctx.position(sym)
            if pos.quantity > 0:
                deployed += pos.quantity * pos.avg_price
                open_count += 1
                sig_exit = self._manage_exit(sym, completed, price, pos, last_ts)
                if sig_exit:
                    signals.append(sig_exit)
                continue

            if not new_bar or market_weak:
                continue  # only new entries on a fresh bar, and only if market is OK

            # --- trend filter (no falling knives) ---
            close = completed["close"]
            sma = float(close.rolling(self.trend_sma).mean().iloc[-1])
            ema = float(close.ewm(span=self.trend_ema, adjust=False).mean().iloc[-1])
            if not (price > sma and price > ema):
                continue

            fundamentals = self._fundamentals(sym) if (live and self.use_fundamentals) else None
            sentiment = None
            if live and self.use_sentiment and ctx.ai is not None:
                try:
                    sentiment = ctx.ai.sentiment_bias(sym)
                except Exception:
                    sentiment = None

            res = self.scorer.score(completed, fundamentals=fundamentals,
                                    sentiment=sentiment, vix=vix)
            if res.signal(self.strong_buy, self.buy) in ("BUY", "STRONG_BUY"):
                candidates.append((res.total, sym, price, completed))

        # --- rank candidates, spend budget, cost-aware gate ---
        remaining = max(0.0, self.capital - deployed)
        cap_per_pos = self.capital * self.max_pos_pct
        for score, sym, price, cdf in sorted(candidates, key=lambda x: x[0], reverse=True):
            if open_count >= self.max_positions or remaining < price or price > cap_per_pos:
                continue
            atr = float(ind.atr(cdf, self.atr_period).iloc[-1])
            strength = score if score >= self.strong_buy else score * 0.5
            qty = position_size(self.capital, price, atr, strength,
                                self.per_trade_risk_pct, self.max_pos_pct)
            qty = min(qty, int(remaining // price))
            if qty < 1:
                continue
            # Improvement 6: only trade if expected move beats round-trip cost.
            expected_profit = self.target_atr_mult * atr * qty
            if expected_profit < self.min_profit_cost_ratio * _round_trip_cost(price, qty):
                continue
            self._state[sym] = {"entry_ts": pd.to_datetime(cdf["date"].iloc[-1]),
                                "peak": price, "atr": atr}
            signals.append(Signal(sym, "BUY", qty,
                                  f"HAASEY v2 picked {sym} (score {score:+.2f}), "
                                  f"qty {qty} @ ~{price:.0f}"))
            remaining -= qty * price
            open_count += 1
        return signals

    # ------------------------------------------------------------------
    def _market_weak(self, ctx) -> bool:
        """True when the index is >regime_tol below its SMA (broad weakness).
        If index data is unavailable, returns False (don't block on missing data)."""
        if not self.use_regime:
            return False
        raw = ctx.bars(self.regime_symbol)
        if len(raw) < 60:
            return False
        ddf = self._decision_df(raw)
        if len(ddf) < self.regime_sma + 1:
            return False
        close = ddf["close"]
        sma = float(close.rolling(self.regime_sma).mean().iloc[-1])
        last = float(close.iloc[-1])
        return last < sma * self.regime_tol

    def _manage_exit(self, sym, cdf, price, pos, last_ts):
        st = self._state.get(sym)
        if st is None:  # e.g. after a restart: seed from current position
            st = {"entry_ts": pd.to_datetime(last_ts), "peak": max(price, pos.avg_price),
                  "atr": float(ind.atr(cdf, self.atr_period).iloc[-1])}
            self._state[sym] = st
        st["peak"] = max(st["peak"], price)
        # Cooldown: cannot exit within the minimum holding period.
        held_min = (pd.to_datetime(last_ts) - st["entry_ts"]).total_seconds() / 60.0
        if held_min < self.cooldown_minutes:
            return None
        trailing_stop = st["peak"] - self.trail_atr_mult * st["atr"]
        if price <= trailing_stop:
            self._state.pop(sym, None)
            return Signal(sym, "SELL", pos.quantity,
                          f"HAASEY v2 trailing stop (peak {st['peak']:.0f}, "
                          f"stop {trailing_stop:.0f})", is_exit=True)
        return None
