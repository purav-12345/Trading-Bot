"""Dynamic position sizing — regime-scaled ATR risk with a fractional-Kelly cap.

Replaces the flat "2% risk per trade" with sizing that:
  * scales by ATR (risk a fixed % of capital per trade),
  * scales by signal conviction (strong signals size up within the cap),
  * scales by REGIME (bear/choppy => smaller, via size_mult),
  * optionally applies FRACTIONAL KELLY when a real edge estimate (win rate +
    payoff ratio) is available — and defaults to conservative when it is not.

Always hard-capped by the risk engine's per-position and per-trade limits, so
this can only ever size *down* relative to those caps, never blow past them.
"""
from __future__ import annotations


def kelly_fraction(win_rate: float, payoff_ratio: float, cap: float = 0.5) -> float:
    """Fractional Kelly f* = W - (1-W)/R, clamped to [0, cap]. Returns 0 if the
    edge is non-positive (don't size up on a losing/unknown edge)."""
    if payoff_ratio <= 0:
        return 0.0
    f = win_rate - (1.0 - win_rate) / payoff_ratio
    return max(0.0, min(f, cap))


def dynamic_size(capital: float, price: float, atr: float, strength: float,
                 per_trade_risk_pct: float, max_pos_pct: float,
                 regime_mult: float = 1.0,
                 win_rate: float | None = None, payoff_ratio: float | None = None,
                 kelly_cap: float = 0.5) -> int:
    """Return share quantity. Base = ATR risk sizing scaled by conviction and
    regime; if a credible edge (win_rate + payoff) is supplied, blend in
    half-fractional-Kelly. Never exceeds the per-position capital cap."""
    if price <= 0:
        return 0
    risk_per_share = max(1.5 * atr, price * 0.001)
    base_risk = capital * per_trade_risk_pct * (0.5 + 0.5 * abs(strength)) * regime_mult

    # Fractional Kelly only when we actually have an edge estimate.
    if win_rate is not None and payoff_ratio is not None:
        f = kelly_fraction(win_rate, payoff_ratio, cap=kelly_cap)
        # Scale risk toward Kelly (0.5 = neutral baseline -> multiply by 2*f).
        base_risk *= max(0.25, min(2.0, 2.0 * f)) if f > 0 else 0.5

    qty = int(base_risk / risk_per_share)
    max_qty = int((capital * max_pos_pct) / price)
    return max(0, min(qty, max_qty))
