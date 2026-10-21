"""LangGraph agent with a human-in-the-loop approval checkpoint.

Flow:   understand -> research -> propose -> [INTERRUPT: human approval] -> commit

The graph pauses at the approval node and will not commit anything until a
human resumes it with approve=True. This is the agentic embodiment of your
"confirm before acting" requirement. The LLM lives ONLY here, in planning —
never in the trade-execution loop.

Import-guarded: if langgraph isn't installed, `build_agent` returns None and
callers fall back to the StrategyPlanner + an explicit approve step (same
human gate, simpler machinery).
"""
from __future__ import annotations

from typing import Any, Optional, TypedDict

from .planner import StrategyPlanner

try:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import interrupt
    _HAS_LANGGRAPH = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_LANGGRAPH = False


class AgentState(TypedDict, total=False):
    instruction: str
    ltp: Optional[float]
    plan_text: str
    config_patch: dict
    warnings: list
    approved: bool
    committed: bool


def build_agent(config, market_data=None, ai=None):
    """Returns a compiled LangGraph app, or None if langgraph isn't available."""
    if not _HAS_LANGGRAPH:
        return None

    planner = StrategyPlanner(ai=ai)

    def understand(state: AgentState) -> AgentState:
        return state  # placeholder for future multi-turn parsing

    def research(state: AgentState) -> AgentState:
        # Pull a live price for context if we can guess the symbol cheaply.
        ltp = None
        if market_data is not None:
            import re
            m = re.findall(r"\b([A-Z]{2,12})\b", state["instruction"])
            if m:
                try:
                    market_data.connect()
                    ltp = market_data.ltp(m[0])
                except Exception:
                    ltp = None
        return {"ltp": ltp}

    def propose(state: AgentState) -> AgentState:
        p = planner.propose(state["instruction"])
        return {"plan_text": p.plan_text, "config_patch": p.config_patch,
                "warnings": p.warnings}

    def approval(state: AgentState) -> AgentState:
        # Pause here for a human decision. Resume with Command(resume={"approved": True}).
        decision = interrupt({
            "plan_text": state.get("plan_text"),
            "config_patch": state.get("config_patch"),
            "warnings": state.get("warnings"),
            "ltp": state.get("ltp"),
        })
        return {"approved": bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)}

    def commit(state: AgentState) -> AgentState:
        if state.get("approved") and state.get("config_patch"):
            planner.apply(config, state["config_patch"])
            return {"committed": True}
        return {"committed": False}

    g = StateGraph(AgentState)
    g.add_node("understand", understand)
    g.add_node("research", research)
    g.add_node("propose", propose)
    g.add_node("approval", approval)
    g.add_node("commit", commit)
    g.add_edge(START, "understand")
    g.add_edge("understand", "research")
    g.add_edge("research", "propose")
    g.add_edge("propose", "approval")
    g.add_edge("approval", "commit")
    g.add_edge("commit", END)
    return g.compile(checkpointer=MemorySaver())
