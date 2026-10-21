"""HAASEY v3 — deep-value, quality-obsessed multi-factor stock picker.

BUY: a 7-factor ensemble (only factors with REAL free data) —
     value (PE/PB/PEG), quality (ROE/D-E), growth (EPS/Revenue),
     stability (current ratio), momentum, technical, sentiment — gated by a
     trend filter and a market-regime filter. (Removed: cashflow bucket,
     promoter holding, ROCE, dividend yield — no free data source for them.)
SELL (hybrid): (1) 4.2% take-profit, (2) ATR trailing stop after cooldown,
     (3) end-of-day square-off handled by the engine.
Keeps v2 anti-churn engineering: 15-min decision bars, cooldown, cost gate,
and the prebucketed fast path for fast multi-month backtests.

HONESTY:
 * Backtest uses NO look-ahead data: fundamentals/news/VIX are point-in-time
   "now" values, so with `ctx.backtest` the fundamental buckets fall back to
   neutral defaults and the score is effectively momentum+technical driven.
 * Even LIVE, the free data source cannot supply ROCE / interest-coverage /
   promoter-holding, so those three filters stay neutral unless you plug in a
   paid/scraped fundamentals source. See src/data/fundamentals.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind
from .base import Signal, Strategy, StrategyContext
from .haasey import _resample, _round_trip_cost


class HaaseyV3Strategy(Strategy):
    type = "haasey_v3"

    def __init__(self, cfg: dict):
        uni = cfg.get("universe", "auto")
        if not uni or uni == "auto":
            from ..data import default_universe
            uni = default_universe()
        cfg.setdefault("symbol", uni[0])
        super().__init__(cfg)

        self.universe = uni
        self.capital = float(cfg.get("capital", 20000))
        self.max_positions = int(cfg.get("max_positions", 5))
        self.per_trade_risk_pct = float(cfg.get("per_trade_risk_pct", 0.015))
        self.max_pos_pct = float(cfg.get("max_pos_pct", 0.20))
        self.atr_period = int(cfg.get("atr_period", 14))

        # v2 structural protections
        self.decision_minutes = int(cfg.get("decision_bar_minutes", 15))
        self.cooldown_minutes = int(cfg.get("cooldown_minutes", 60))
        self.trail_atr_mult = float(cfg.get("trail_atr_mult", 2.5))
        self.target_atr_mult = float(cfg.get("target_atr_mult", 3.0))
        self.min_profit_cost_ratio = float(cfg.get("min_profit_cost_ratio", 1.6))
        self.trend_sma = int(cfg.get("trend_sma", 50))
        self.trend_ema = int(cfg.get("trend_ema", 20))
        # Multi-timeframe confluence: only enter when the 15m signal agrees with
        # the higher-timeframe (e.g. 60m) trend — filters counter-trend entries.
        self.use_mtf = bool(cfg.get("use_multitimeframe", True))
        self.htf_minutes = int(cfg.get("htf_minutes", 60))
        self.htf_sma = int(cfg.get("htf_sma", 20))

        # regime filter (3-state: BULL/BEAR/CHOPPY from index trend + VIX)
        self.use_regime = bool(cfg.get("use_regime", True))
        self.regime_symbol = cfg.get("regime_symbol", "NIFTY 50")
        self.regime_sma = int(cfg.get("regime_sma", 50))
        self.regime_vix_high = float(cfg.get("regime_vix_high", 20.0))
        self._last_regime = None  # last RegimeState (for logging + sizing)

        # v3 SELL: user's take-profit
        self.take_profit_pct = float(cfg.get("take_profit_pct", 0.042))

        # v3 BUY weights (sum = 1.0). Removed factors with no free data:
        # cashflow (FCF + interest-coverage), promoter, ROCE, dividend yield.
        # Remaining live factors reweighted to fill the gap.
        self.w_value = float(cfg.get("w_value", 0.25))       # PE, PB, PEG
        self.w_quality = float(cfg.get("w_quality", 0.22))   # ROE, D/E
        self.w_growth = float(cfg.get("w_growth", 0.22))     # EPS, Revenue growth
        self.w_stability = float(cfg.get("w_stability", 0.05))  # Current ratio only
        self.w_momentum = float(cfg.get("w_momentum", 0.13))
        self.w_technical = float(cfg.get("w_technical", 0.08))
        self.w_sentiment = float(cfg.get("w_sentiment", 0.05))

        # Regime-aware LIVE weight sets (momentum-led in BULL, value/quality in
        # BEAR). Only used live (backtest freezes fundamentals). Each sums ~1.0.
        self.use_regime_weights = bool(cfg.get("use_regime_weights", True))
        self._regime_weights = {
            "BULL":   {"value": 0.15, "quality": 0.20, "growth": 0.15, "stability": 0.05,
                       "momentum": 0.25, "technical": 0.15, "sentiment": 0.05},
            "BEAR":   {"value": 0.30, "quality": 0.25, "growth": 0.10, "stability": 0.15,
                       "momentum": 0.05, "technical": 0.05, "sentiment": 0.10},
            "CHOPPY": {"value": 0.25, "quality": 0.22, "growth": 0.15, "stability": 0.08,
                       "momentum": 0.10, "technical": 0.15, "sentiment": 0.05},
        }
        # Time-of-day windows (IST): only enter inside these (skip midday lull).
        self.trade_windows = self._parse_windows(cfg.get("trade_windows", ""))

        self.strong_buy = float(cfg.get("strong_buy", 0.65))
        self.buy = float(cfg.get("buy", 0.50))
        self.use_fundamentals = bool(cfg.get("use_fundamentals", True))
        self.use_sentiment = bool(cfg.get("use_sentiment", True))
        self.use_macro = bool(cfg.get("use_macro", True))
        self.use_finbert = bool(cfg.get("use_finbert", True))  # prefer FinBERT over Gemini
        self._finbert = None

        self.scorer = None  # v3 scores inline
        self._fund = None
        self._prebucketed = False
        self._state: dict[str, dict] = {}
        self._last_bar: dict[str, object] = {}
        # closed-trade record (drives fractional-Kelly sizing)
        self._closed = 0
        self._wins = 0
        self._gross_win = 0.0
        self._gross_loss = 0.0

    def _record_close(self, entry_price: float, exit_price: float, qty: int) -> None:
        pnl = (exit_price - entry_price) * qty
        self._closed += 1
        if pnl >= 0:
            self._wins += 1
            self._gross_win += pnl
        else:
            self._gross_loss += -pnl

    # -- engine hooks --
    def set_prebucketed(self, flag: bool = True) -> None:
        self._prebucketed = flag

    def _decision_df(self, raw):
        if self._prebucketed:
            return raw
        rdf = _resample(raw, self.decision_minutes)
        return rdf.iloc[:-1] if len(rdf) > 1 else rdf

    def _htf_ok(self, raw) -> bool:
        """Higher-timeframe confluence: close above the HTF SMA (uptrend).
        Returns True (don't block) if there isn't enough HTF data yet."""
        htf = _resample(raw, self.htf_minutes)
        if len(htf) < self.htf_sma + 1:
            return True
        close = htf["close"]
        sma = float(close.rolling(self.htf_sma).mean().iloc[-1])
        return float(close.iloc[-1]) >= sma

    def _fundamentals(self, symbol):
        if self._fund is None:
            from ..data import FundamentalsProvider
            self._fund = FundamentalsProvider()
        return self._fund.get(symbol)

    # ------------------------------------------------------------------
    def _weights(self, backtest: bool) -> dict:
        """Dual-mode weights (7 factors; cashflow bucket removed).

        BACKTEST: technicals/momentum drive (~74%) so the trend-following edge is
        actually testable; fundamentals are frozen at neutral in backtest anyway.
        LIVE: the fundamental factors that have REAL free data lead.
        """
        if backtest:
            return {"value": 0.10, "quality": 0.08, "growth": 0.06,
                    "stability": 0.01, "momentum": 0.23, "technical": 0.51, "sentiment": 0.01}
        # LIVE: regime-aware weight set if enabled and regime known.
        if self.use_regime_weights and self._last_regime is not None:
            rw = self._regime_weights.get(self._last_regime.label)
            if rw:
                return dict(rw)
        return {"value": self.w_value, "quality": self.w_quality,
                "growth": self.w_growth, "stability": self.w_stability,
                "momentum": self.w_momentum, "technical": self.w_technical,
                "sentiment": self.w_sentiment}

    @staticmethod
    def _combine(s: dict, w: dict) -> float:
        return float(np.clip(sum(w[k] * s[k] for k in w), 0, 1))

    @staticmethod
    def _parse_windows(spec: str):
        """'09:30-11:30,13:30-15:00' -> [(570,690),(810,900)] minutes-of-day."""
        out = []
        for part in (spec or "").split(","):
            part = part.strip()
            if "-" not in part:
                continue
            a, b = part.split("-")
            ah, am = a.split(":"); bh, bm = b.split(":")
            out.append((int(ah) * 60 + int(am), int(bh) * 60 + int(bm)))
        return out

    def _in_trade_window(self, now) -> bool:
        if not self.trade_windows:
            return True
        m = now.hour * 60 + now.minute
        return any(s <= m <= e for s, e in self.trade_windows)

    def _score_stock(self, df: pd.DataFrame, sym: str, live: bool, vix, backtest: bool = False) -> dict:
        price = float(df["close"].iloc[-1])
        close = df["close"]
        f = self._fundamentals(sym) if (live and self.use_fundamentals) else {}
        f = f or {}

        # 1) VALUE
        pe, pb, peg = f.get("trailingPE", 20), f.get("priceToBook", 3.0), f.get("pegRatio", 1.5)
        value = (np.clip(1 - (pe - 5) / 45, 0, 1) * 0.4
                 + np.clip(1 - (pb - 0.5) / 5.5, 0, 1) * 0.3
                 + np.clip(1 - (peg - 0.2) / 2.8, 0, 1) * 0.3)
        # 2) QUALITY (ROE + D/E; ROCE removed — no free data)
        roe, de = f.get("returnOnEquity", 12), f.get("debtToEquity", 0.5)
        quality = (np.clip((roe - 5) / 25, 0, 1) * 0.6
                   + np.clip(1 - (de - 0.1) / 1.9, 0, 1) * 0.4)
        # 3) GROWTH
        eg, rg = f.get("earningsGrowth", 8), f.get("revenueGrowth", 10)
        growth = np.clip((eg + 10) / 40, 0, 1) * 0.6 + np.clip((rg + 10) / 40, 0, 1) * 0.4
        # 4) STABILITY (current ratio only; promoter + dividend removed)
        cr = f.get("currentRatio", 1.5)
        stability = float(np.clip((cr - 0.8) / 2.2, 0, 1))
        # (removed: cashflow bucket [FCF, interest-coverage], promoter, ROCE, dividend)
        # 6) MOMENTUM
        ret_1m = (price / float(close.iloc[-20]) - 1) * 100 if len(close) >= 20 else 0.0
        high_52w = f.get("fiftyTwoWeekHigh", price * 1.2) or price * 1.2
        momentum = np.clip((ret_1m + 10) / 25, 0, 1) * 0.6 + np.clip(price / high_52w, 0, 1) * 0.4
        # 7) TECHNICAL
        rsi = float(ind.rsi(close, 14).iloc[-1])
        macd = float(ind.macd(df).iloc[-1])
        technical = np.clip(1 - (rsi - 30) / 40, 0, 1) * 0.6 + (1.0 if macd > 0 else 0.0) * 0.4
        # 8) SENTIMENT (neutral unless live AI overrides)
        sentiment = 0.5

        s = {"value": value, "quality": quality, "growth": growth,
             "stability": stability, "momentum": momentum, "technical": technical,
             "sentiment": sentiment, "pe": pe, "roe": roe}
        s["total"] = self._combine(s, self._weights(backtest))
        if vix and vix > 25:  # macro fear penalty
            s["total"] *= max(0.7, 1.0 - (vix - 25) / 50)
        return s

    # ------------------------------------------------------------------
    def on_tick(self, ctx: StrategyContext) -> list[Signal]:
        live = not ctx.backtest

        # VIX first (regime uses it), live-only.
        vix = None
        if live and self.use_macro:
            try:
                from ..data import india_vix
                vix = india_vix()
            except Exception:
                vix = None

        # 3-state regime: BULL / BEAR / CHOPPY. BEAR blocks NEW entries; the
        # size_mult scales position size (used by _calc_size). Existing
        # positions are always still managed/exited below.
        market_weak = False
        size_mult = 1.0
        if self.use_regime:
            from ..data import classify_regime
            nifty = ctx.bars(self.regime_symbol)
            ndf = self._decision_df(nifty) if len(nifty) >= 60 else None
            regime = classify_regime(ndf, vix=vix, sma_slow=self.regime_sma,
                                     vix_high=self.regime_vix_high)
            self._last_regime = regime
            market_weak = not regime.allow_new_entries
            size_mult = regime.size_mult

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

            # ---- EXIT: 4.2% take-profit, then ATR trailing stop after cooldown ----
            if pos.quantity > 0:
                deployed += pos.quantity * pos.avg_price
                open_count += 1
                st = self._state.get(sym)
                if st is None:
                    st = {"entry_ts": pd.to_datetime(last_ts),
                          "peak": max(price, pos.avg_price),
                          "atr": float(ind.atr(completed, self.atr_period).iloc[-1]),
                          "entry_price": pos.avg_price}
                    self._state[sym] = st
                st["peak"] = max(st["peak"], price)

                tp_price = st["entry_price"] * (1 + self.take_profit_pct)
                if price >= tp_price:
                    self._record_close(st["entry_price"], price, pos.quantity)
                    self._state.pop(sym, None)
                    signals.append(Signal(sym, "SELL", pos.quantity,
                                          f"take-profit {self.take_profit_pct*100:.1f}% @ {tp_price:.2f}",
                                          is_exit=True))
                    continue
                held_min = (pd.to_datetime(last_ts) - st["entry_ts"]).total_seconds() / 60.0
                if held_min >= self.cooldown_minutes:
                    stop = st["peak"] - self.trail_atr_mult * st["atr"]
                    if price <= stop:
                        self._record_close(st["entry_price"], price, pos.quantity)
                        self._state.pop(sym, None)
                        signals.append(Signal(sym, "SELL", pos.quantity,
                                              f"trailing stop (peak {st['peak']:.0f}, stop {stop:.0f})",
                                              is_exit=True))
                continue

            # ---- ENTRY (flat, new bar, market OK, in trade window, uptrend) ----
            if not new_bar or market_weak:
                continue
            if not self._in_trade_window(ctx.now):
                continue  # skip midday lull / open-close edges
            close = completed["close"]
            sma = float(close.rolling(self.trend_sma).mean().iloc[-1])
            ema = float(close.ewm(span=self.trend_ema, adjust=False).mean().iloc[-1])
            if not (price > sma and price > ema):
                continue
            # Multi-timeframe confluence: require the higher timeframe to agree.
            if self.use_mtf and not self._htf_ok(raw):
                continue

            s = self._score_stock(completed, sym, live, vix, backtest=ctx.backtest)
            if live and self.use_sentiment:
                bias = None
                # Prefer FinBERT (finance-specific); fall back to Gemini.
                if self.use_finbert:
                    if self._finbert is None:
                        from ..ai.finbert import FinBertSentiment
                        self._finbert = FinBertSentiment()
                    if self._finbert.available:
                        bias = self._finbert.sentiment_bias(sym)
                if bias is None and ctx.ai is not None:
                    try:
                        bias = float(ctx.ai.sentiment_bias(sym))
                    except Exception:
                        bias = None
                if bias is not None:
                    s["sentiment"] = (bias + 1) / 2            # -1/0/+1 -> 0/0.5/1
                    s["total"] = self._combine(s, self._weights(ctx.backtest))
            if s["total"] >= self.buy:
                candidates.append((s["total"], sym, price, completed, s))

        # ---- allocate budget across ranked candidates ----
        remaining = max(0.0, self.capital - deployed)
        cap_per_pos = self.capital * self.max_pos_pct
        for score, sym, price, cdf, s in sorted(candidates, key=lambda x: x[0], reverse=True):
            if open_count >= self.max_positions or remaining < price or price > cap_per_pos:
                continue
            atr = float(ind.atr(cdf, self.atr_period).iloc[-1])
            strength = score if score >= self.strong_buy else score * 0.5
            qty = self._calc_size(price, atr, strength, size_mult)
            qty = min(qty, int(remaining // price))
            if qty < 1:
                continue
            if self.target_atr_mult * atr * qty < self.min_profit_cost_ratio * _round_trip_cost(price, qty):
                continue
            self._state[sym] = {"entry_ts": pd.to_datetime(cdf["date"].iloc[-1]),
                                "peak": price, "atr": atr, "entry_price": price}
            reg = self._last_regime.label if self._last_regime else "n/a"
            signals.append(Signal(sym, "BUY", qty,
                                  f"v3[{reg}] {sym} S:{score:.2f} PE:{s['pe']:.1f} ROE:{s['roe']:.0f}%"))
            remaining -= qty * price
            open_count += 1
        return signals

    def _calc_size(self, price, atr, strength, size_mult: float = 1.0):
        from ..risk import dynamic_size
        # Edge estimate from this strategy's own closed trades so far (fractional
        # Kelly). None until we have enough round-trips -> conservative default.
        win_rate = payoff = None
        if self._closed >= 10:
            win_rate = self._wins / self._closed
            payoff = (self._gross_win / self._wins) / (self._gross_loss / max(1, self._closed - self._wins)) \
                if self._wins and (self._closed - self._wins) and self._gross_loss else None
        return dynamic_size(self.capital, price, atr, strength,
                            self.per_trade_risk_pct, self.max_pos_pct,
                            regime_mult=size_mult, win_rate=win_rate, payoff_ratio=payoff)
