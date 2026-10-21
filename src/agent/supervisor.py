"""LangGraph strategy supervisor — regime-aware, human-gated adaptivity.

Graph:  detect_regime -> propose (regime-appropriate config tilt)
        -> [HUMAN APPROVAL interrupt] -> commit

Keeps the adaptive "brain" LangGraph-orchestrated and human-gated, while the
per-tick execution stays pure/deterministic (this never places orders). On
approval it writes a regime-appropriate parameter tilt to settings.yaml via the
same StrategyPlanner.apply path used by the chat agent.

Import-guarded: if langgraph isn't installed, `run_supervisor` falls back to a
propose-only flow (still human-gated by the --approve flag).
"""
from __future__ import annotations

from typing import Optional, TypedDict

from ..config import Config
from .graph import _HAS_LANGGRAPH

# Regime-appropriate parameter tilts (proposed; never auto-applied).
REGIME_TILTS = {
    "BULL":   {"buy": 0.50, "w_momentum": 0.18, "w_technical": 0.10, "note": "trend-follow: momentum up"},
    "CHOPPY": {"buy": 0.60, "w_momentum": 0.08, "w_value": 0.28, "note": "range: value up, stricter entry"},
    "BEAR":   {"buy": 0.70, "w_stability": 0.12, "note": "defensive: high threshold, quality tilt"},
    "UNKNOWN": {"note": "insufficient index data — no change proposed"},
}


def detect_regime(config: Config):
    """Classify the current live regime from NIFTY + VIX."""
    from ..data import classify_regime, india_vix
    from ..market import MarketData
    from ..strategies.haasey import _resample

    md = MarketData(config)
    md.connect()
    try:
        raw = md.historical("NIFTY 50", interval="minute", days=30)
        ndf = _resample(raw, 15)
    except Exception:
        ndf = None
    vix = None
    try:
        vix = india_vix()
    except Exception:
        pass
    return classify_regime(ndf, vix=vix, sma_slow=50), vix


def build_proposal(config: Config) -> dict:
    regime, vix = detect_regime(config)
    tilt = dict(REGIME_TILTS.get(regime.label, REGIME_TILTS["UNKNOWN"]))
    note = tilt.pop("note", "")
    plan = (f"Detected regime: {regime.label} (vix={vix}, {regime.detail}).\n"
            f"  Proposed HAASEY tilt: {tilt or '(none)'}\n"
            f"  Rationale: {note}")
    return {"regime": regime.label, "tilt": tilt, "plan_text": plan}


def _apply_tilt(config: Config, tilt: dict) -> None:
    if not tilt:
        return
    for s in config.settings.get("strategies", []):
        if s.get("type") == "haasey_v3":
            s.update(tilt)
    config.save()


class SupState(TypedDict, total=False):
    regime: str
    tilt: dict
    plan_text: str
    approved: bool
    committed: bool


def run_supervisor(config: Config, approve: bool = False) -> dict:
    """Detect regime -> propose tilt -> (human gate) -> commit. Single process:
    the --approve flag IS the human approval. With LangGraph installed this runs
    through the interrupt graph; otherwise it uses the same propose/commit logic.
    """
    proposal = build_proposal(config)
    print(proposal["plan_text"])

    if _HAS_LANGGRAPH:
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.graph import END, START, StateGraph
        from langgraph.types import Command, interrupt

        def propose(state: SupState) -> SupState:
            return proposal

        def approval(state: SupState) -> SupState:
            decision = interrupt({"plan_text": state["plan_text"], "tilt": state["tilt"]})
            return {"approved": bool(decision.get("approved") if isinstance(decision, dict) else decision)}

        def commit(state: SupState) -> SupState:
            if state.get("approved") and state.get("tilt"):
                _apply_tilt(config, state["tilt"])
                return {"committed": True}
            return {"committed": False}

        g = StateGraph(SupState)
        g.add_node("propose", propose)
        g.add_node("approval", approval)
        g.add_node("commit", commit)
        g.add_edge(START, "propose")
        g.add_edge("propose", "approval")
        g.add_edge("approval", "commit")
        g.add_edge("commit", END)
        app = g.compile(checkpointer=MemorySaver())
        cfg = {"configurable": {"thread_id": "supervisor-1"}}
        app.invoke({}, cfg)  # pauses at the approval interrupt
        if not approve:
            return {"regime": proposal["regime"], "committed": False,
                    "message": "paused for approval — re-run with --approve to commit"}
        done = app.invoke(Command(resume={"approved": True}), cfg)
        return {"regime": proposal["regime"], "committed": bool(done.get("committed"))}

    # Fallback (no langgraph): the --approve flag is the human gate.
    if approve:
        _apply_tilt(config, proposal["tilt"])
        return {"regime": proposal["regime"], "committed": True}
    return {"regime": proposal["regime"], "committed": False,
            "message": "propose-only — re-run with --approve to commit"}
