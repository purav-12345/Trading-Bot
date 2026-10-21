"""Turns a plain-English instruction into a proposed config change.

  "trade GAIL, buy 500 at 150 sell at 155, stop at 30000 profit or 2000 loss"
      -> {risk:{daily_profit_target:30000, daily_max_loss:2000},
          strategies:[{type:threshold, symbol:GAIL, ...}]}

Two backends, in order:
  1. Gemini (Flash-Lite) for robust natural-language understanding.
  2. A regex heuristic fallback so it still works offline / without a key.

It only PROPOSES. Applying the change is a separate, explicit step (the
human-approval gate). The LLM never touches the live trading loop.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Proposal:
    plan_text: str
    config_patch: dict[str, Any]
    source: str = "heuristic"          # "gemini" or "heuristic"
    warnings: list[str] = field(default_factory=list)


_NUM = r"(\d[\d,]*\.?\d*)"


def _num(s: str) -> float:
    return float(s.replace(",", ""))


class StrategyPlanner:
    def __init__(self, ai=None):
        self.ai = ai  # GeminiHelper or None

    # ------------------------------------------------------------------
    def propose(self, instruction: str) -> Proposal:
        if self.ai is not None and self.ai.available:
            p = self._propose_gemini(instruction)
            if p is not None:
                return p
        return self._propose_heuristic(instruction)

    # ------------------------------------------------------------------
    def _propose_gemini(self, instruction: str) -> Optional[Proposal]:
        schema_hint = (
            "Return ONLY a JSON object, no prose, with this shape:\n"
            "{\n"
            '  "risk": {"daily_profit_target": <int|null>, "daily_max_loss": <int|null>},\n'
            '  "strategy": {\n'
            '     "type": "threshold" | "supertrend_orb" | "ma_crossover",\n'
            '     "symbol": "<NSE tradingsymbol>",\n'
            '     "quantity": <int>,\n'
            '     "entry_price": <float|null>, "target_price": <float|null>,\n'
            '     "stoploss_price": <float|null>, "re_enter": <bool>\n'
            "  }\n"
            "}\n"
            "Use type 'threshold' when the user gives explicit buy/sell prices. "
            "Use 'supertrend_orb' for autonomous momentum. Omit unknown fields as null."
        )
        out = self.ai.reason(
            f"User instruction:\n{instruction}\n\n{schema_hint}",
            system="You convert trading instructions into strict JSON config. Output JSON only.",
        )
        if not out:
            return None
        data = _extract_json(out)
        if not data:
            return None
        patch = self._normalize(data)
        return Proposal(self._describe(patch), patch, source="gemini",
                        warnings=_sanity_warnings(patch))

    # ------------------------------------------------------------------
    def _propose_heuristic(self, instruction: str) -> Proposal:
        text = instruction.lower()
        risk: dict[str, Any] = {}

        # Accept both "30000 profit" and "profit 30000" — number-before-keyword first.
        m = re.search(rf"{_NUM}[^0-9a-z]*profit", text) or re.search(rf"profit[^0-9]*{_NUM}", text)
        if m:
            risk["daily_profit_target"] = int(_num(m.group(1)))
        m = re.search(rf"{_NUM}[^0-9a-z]*loss", text) or re.search(rf"loss[^0-9]*{_NUM}", text)
        if m:
            risk["daily_max_loss"] = int(_num(m.group(1)))

        strat: dict[str, Any] = {"re_enter": True}
        # symbol: any ALL-CAPS token in the original, else word after "trade".
        stop_caps = {"BUY", "SELL", "RS", "INR", "AND", "OR", "THE", "AT", "QTY",
                     "SL", "STOP", "LOSS", "PROFIT"}
        caps = [c for c in re.findall(r"\b([A-Z]{2,12})\b", instruction) if c not in stop_caps]
        if caps:
            sym = caps[0]
        else:
            mw = re.search(r"trade\s+([a-zA-Z]{2,12})", text)
            sym = mw.group(1).upper() if mw else None
        if sym:
            strat["symbol"] = sym

        # "qty 500" / "quantity 500" (keyword first) or "500 shares" (number first).
        mq = re.search(rf"(?:qty|quantity)\s*{_NUM}", text) or re.search(rf"{_NUM}\s*shares?", text)
        if mq:
            strat["quantity"] = int(_num(mq.group(1)))

        mb = re.search(rf"buy[^0-9]*(?:at\s*)?{_NUM}", text)
        ms = re.search(rf"sell[^0-9]*(?:at\s*)?{_NUM}", text)
        msl = re.search(rf"(?:stoploss|stop loss|sl)[^0-9]*{_NUM}", text)
        if mb:
            strat["entry_price"] = _num(mb.group(1))
        if ms:
            strat["target_price"] = _num(ms.group(1))
        if msl:
            strat["stoploss_price"] = _num(msl.group(1))

        if "entry_price" in strat and "target_price" in strat:
            strat["type"] = "threshold"
        elif "symbol" in strat:
            strat["type"] = "supertrend_orb"

        patch: dict[str, Any] = {}
        if risk:
            patch["risk"] = risk
        if strat.get("type"):
            patch["strategy"] = strat
        return Proposal(self._describe(patch), patch, source="heuristic",
                        warnings=_sanity_warnings(patch))

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(data: dict) -> dict:
        patch: dict[str, Any] = {}
        risk = {k: v for k, v in (data.get("risk") or {}).items() if v is not None}
        if risk:
            patch["risk"] = risk
        strat = {k: v for k, v in (data.get("strategy") or {}).items() if v is not None}
        if strat.get("type"):
            patch["strategy"] = strat
        return patch

    @staticmethod
    def _describe(patch: dict) -> str:
        lines = ["Here is what I'll set (nothing is live until you approve):"]
        if "risk" in patch:
            r = patch["risk"]
            if "daily_profit_target" in r:
                lines.append(f"  • Stop the day at +Rs.{r['daily_profit_target']:,} profit")
            if "daily_max_loss" in r:
                lines.append(f"  • Hard-stop & lock the day at -Rs.{r['daily_max_loss']:,} loss")
        s = patch.get("strategy")
        if s:
            lines.append(f"  • Strategy: {s.get('type')} on {s.get('symbol', '?')} "
                         f"(qty {s.get('quantity', '?')})")
            if s.get("type") == "threshold":
                lines.append(f"      buy<= {s.get('entry_price')}, sell>= {s.get('target_price')}, "
                             f"SL {s.get('stoploss_price', 'none')}, re-enter {s.get('re_enter', True)}")
        if not patch:
            lines.append("  • (couldn't extract anything actionable — try being explicit, "
                         "e.g. 'trade GAIL buy 150 sell 155 qty 500, stop 30000 profit 2000 loss')")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def apply(self, config, patch: dict, strategy_name: Optional[str] = None) -> None:
        """Commit an APPROVED patch into settings (risk limits + add strategy)."""
        if "risk" in patch:
            config.settings.setdefault("risk", {}).update(patch["risk"])
        if "strategy" in patch:
            strat = dict(patch["strategy"])
            strat.setdefault("name", strategy_name or f"{strat.get('type')}_{strat.get('symbol','x')}".lower())
            strat["enabled"] = True
            existing = config.settings.setdefault("strategies", [])
            existing[:] = [s for s in existing if s.get("name") != strat["name"]]
            existing.append(strat)
        config.save()


def _extract_json(text: str) -> Optional[dict]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None


def _sanity_warnings(patch: dict) -> list[str]:
    warns = []
    s = patch.get("strategy", {})
    if s.get("type") == "threshold":
        e, t = s.get("entry_price"), s.get("target_price")
        if e and t and t <= e:
            warns.append("target_price <= entry_price: this would never book a profit.")
        sl = s.get("stoploss_price")
        if e and sl and sl >= e:
            warns.append("stoploss_price >= entry_price: stop is above entry for a long.")
    r = patch.get("risk", {})
    if r.get("daily_max_loss", 1) <= 0:
        warns.append("daily_max_loss should be a positive rupee amount.")
    return warns
