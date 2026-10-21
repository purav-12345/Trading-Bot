"""FastAPI dashboard + chat/approval API.

Endpoints:
  GET  /                 -> dashboard (status, P&L, positions, lock state)
  GET  /api/status       -> JSON status
  POST /api/chat         -> {instruction} -> proposed config (NOT applied)
  POST /api/approve      -> {patch} -> applies an approved proposal
  POST /api/run          -> start a background paper session (sandbox)
  POST /api/stop         -> manual kill switch (flatten + lock)
  POST /api/reset        -> unlock the day

The chat -> approve split IS the human-in-the-loop gate: the bot proposes,
you click Approve, only then does config change. Nothing trades without a
running session, and the risk engine still governs every order.
"""
from __future__ import annotations

import threading

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from ..agent import StrategyPlanner
from ..ai import GeminiHelper
from ..config import load_config
from ..engine import Trader

app = FastAPI(title="TradingBot")

_session_thread: threading.Thread | None = None
_TRADER: Trader | None = None
_LOCK = threading.Lock()


def _trader() -> Trader:
    """One shared Trader so the dashboard reflects the live (paper/real)
    session's positions and P&L instead of a fresh, empty broker each call."""
    global _TRADER
    with _LOCK:
        if _TRADER is None:
            _TRADER = Trader(load_config())
            _TRADER.connect()
        return _TRADER


def _reload_trader() -> Trader:
    """Rebuild after a config change (e.g. an approved chat proposal)."""
    global _TRADER
    with _LOCK:
        _TRADER = Trader(load_config())
        _TRADER.connect()
        return _TRADER


class ChatIn(BaseModel):
    instruction: str


class ApproveIn(BaseModel):
    patch: dict
    name: str | None = None


@app.get("/api/status")
def api_status():
    return _trader().status()


@app.post("/api/chat")
def api_chat(body: ChatIn):
    cfg = load_config()
    ai = GeminiHelper(cfg) if cfg.ai.get("enabled") else None
    planner = StrategyPlanner(ai=ai)
    p = planner.propose(body.instruction)
    return {"plan_text": p.plan_text, "config_patch": p.config_patch,
            "warnings": p.warnings, "source": p.source}


@app.post("/api/approve")
def api_approve(body: ApproveIn):
    cfg = load_config()
    StrategyPlanner().apply(cfg, body.patch, strategy_name=body.name)
    _reload_trader()  # pick up the new strategy/limits immediately
    return {"ok": True, "message": "Applied. Active on the next start."}


@app.post("/api/run")
def api_run():
    global _session_thread
    if _session_thread and _session_thread.is_alive():
        return {"ok": False, "message": "A session is already running."}

    def _go():
        _trader().run_session(poll_seconds=2.0, respect_session=False, max_seconds=300)

    _session_thread = threading.Thread(target=_go, daemon=True)
    _session_thread.start()
    return {"ok": True, "message": "Paper session started (sandbox, 5 min)."}


@app.post("/api/stop")
def api_stop():
    t = _trader()
    t.risk.manual_stop("manual stop via web", t.notifier)
    return {"ok": True, "message": "Stopped: squared off + locked."}


@app.post("/api/reset")
def api_reset():
    _trader().risk.reset_day()
    return {"ok": True, "message": "Day unlocked."}


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return _HTML


_HTML = """
<!doctype html><html><head><meta charset="utf-8"><title>TradingBot</title>
<style>
 body{font-family:system-ui,Segoe UI,Arial;margin:0;background:#0f172a;color:#e2e8f0}
 .wrap{max-width:860px;margin:0 auto;padding:24px}
 h1{font-size:20px} .card{background:#1e293b;border-radius:12px;padding:16px;margin:12px 0}
 .pnl{font-size:34px;font-weight:700} .pos{color:#4ade80}.neg{color:#f87171}
 button{background:#3b82f6;color:#fff;border:0;border-radius:8px;padding:10px 14px;margin:4px;cursor:pointer}
 button.stop{background:#ef4444} button.reset{background:#64748b}
 input,textarea{width:100%;box-sizing:border-box;background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:8px;padding:10px}
 .badge{display:inline-block;padding:3px 10px;border-radius:999px;font-size:12px}
 .locked{background:#7f1d1d}.active{background:#14532d}
 pre{white-space:pre-wrap;background:#0f172a;padding:10px;border-radius:8px}
 small{color:#94a3b8}
</style></head><body><div class="wrap">
 <h1>🤖 TradingBot <small id="mode"></small></h1>
 <div class="card">
   <div>Today's P&L <span id="badge" class="badge"></span></div>
   <div class="pnl" id="pnl">—</div>
   <small id="limits"></small>
   <div id="positions"></div>
   <div style="margin-top:10px">
     <button onclick="act('run')">▶ Start (paper)</button>
     <button class="stop" onclick="act('stop')">🛑 Stop now</button>
     <button class="reset" onclick="act('reset')">↺ Reset / Unlock</button>
   </div>
 </div>
 <div class="card">
   <h3>💬 Tell the bot what to do</h3>
   <textarea id="instr" rows="3" placeholder="trade GAIL buy 150 sell 155 qty 500, stop at 30000 profit 2000 loss"></textarea>
   <button onclick="chat()">Propose plan</button>
   <div id="proposal"></div>
 </div>
 <small>Paper-first. Live trading only when TRADING_MODE=live. The risk engine governs every order.</small>
</div>
<script>
let lastPatch=null;
async function refresh(){
  const s=await (await fetch('/api/status')).json();
  document.getElementById('mode').textContent='· '+s.mode+(s.live_data?' · live data':' · synthetic data');
  const p=s.total_pnl;
  const el=document.getElementById('pnl'); el.textContent='₹'+p.toLocaleString();
  el.className='pnl '+(p>=0?'pos':'neg');
  const b=document.getElementById('badge'); b.textContent=s.status; b.className='badge '+(s.locked?'locked':'active');
  document.getElementById('limits').textContent='Target +₹'+s.profit_target.toLocaleString()+' · Max loss -₹'+s.max_loss.toLocaleString()+' · trades '+s.trade_count+(s.lock_reason?(' · '+s.lock_reason):'');
  const pos=Object.entries(s.positions||{}).map(([k,v])=>k+': '+v.qty+' @ '+v.avg+' (uPnL ₹'+v.uPnL+')').join('<br>');
  document.getElementById('positions').innerHTML=pos?('<pre>'+pos+'</pre>'):'';
}
async function act(kind){ const r=await (await fetch('/api/'+kind,{method:'POST'})).json(); alert(r.message||JSON.stringify(r)); refresh(); }
async function chat(){
  const instruction=document.getElementById('instr').value;
  const r=await (await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({instruction})})).json();
  lastPatch=r.config_patch;
  let html='<pre>'+r.plan_text+'</pre>';
  if(r.warnings&&r.warnings.length) html+='<pre>⚠️ '+r.warnings.join('\\n⚠️ ')+'</pre>';
  if(r.config_patch&&Object.keys(r.config_patch).length) html+='<button onclick="approve()">✅ Approve & apply</button>';
  html+='<br><small>source: '+r.source+'</small>';
  document.getElementById('proposal').innerHTML=html;
}
async function approve(){
  const r=await (await fetch('/api/approve',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({patch:lastPatch})})).json();
  alert(r.message); refresh();
}
refresh(); setInterval(refresh,3000);
</script></body></html>
"""
