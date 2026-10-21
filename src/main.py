"""Command-line entrypoint.

  python -m src.main run                  # paper session, respects market hours
  python -m src.main run --sandbox 60     # ignore market hours, run 60s (paper test)
  python -m src.main daemon               # auto start/stop every trading day
  python -m src.main status               # show P&L / lock state / positions
  python -m src.main reset                # unlock the day ("press Run again")
  python -m src.main stop                 # manual kill switch: flatten + lock now
  python -m src.main chat "trade GAIL buy 150 sell 155 qty 500, stop 30000 profit 2000 loss"
  python -m src.main chat "..." --yes     # approve & apply the proposed config
  python -m src.main backtest             # backtest enabled strategies
  python -m src.main kite-login           # step 1: print Zerodha login URL
  python -m src.main kite-login --request-token TOKEN   # step 2: get access token
  python -m src.main web                  # launch dashboard + chat at :8000
"""
from __future__ import annotations

import argparse
import json
import sys

from .agent import StrategyPlanner
from .ai import GeminiHelper
from .config import load_config


def _print(obj):
    print(json.dumps(obj, indent=2, default=str))


def cmd_run(args):
    from .engine import Trader

    cfg = load_config()
    trader = Trader(cfg)
    if args.sandbox is not None:
        trader.run_session(poll_seconds=args.poll, respect_session=False,
                           max_seconds=float(args.sandbox))
    else:
        trader.run_session(poll_seconds=args.poll, respect_session=True)


def cmd_daemon(args):
    from .engine.scheduler import run_daemon

    run_daemon(load_config(), poll_seconds=args.poll)


def cmd_status(args):
    from .engine import Trader

    trader = Trader(load_config())
    trader.connect()
    _print(trader.status())


def cmd_reset(args):
    from .engine import Trader

    trader = Trader(load_config())
    trader.risk.reset_day()
    print("✅ Day unlocked. The bot will trade again on the next run.")


def cmd_stop(args):
    from .engine import Trader

    trader = Trader(load_config())
    trader.connect()
    trader.risk.manual_stop("manual stop via CLI", trader.notifier)
    print("🛑 Manual stop: positions squared off and day locked.")


def cmd_chat(args):
    cfg = load_config()
    ai = GeminiHelper(cfg) if cfg.ai.get("enabled") else None
    planner = StrategyPlanner(ai=ai)
    proposal = planner.propose(args.instruction)
    print(f"\n🤖 ({proposal.source}) proposed plan:\n{proposal.plan_text}\n")
    for w in proposal.warnings:
        print(f"   ⚠️  {w}")
    if not proposal.config_patch:
        return
    if args.yes:
        planner.apply(cfg, proposal.config_patch)
        print("\n✅ Approved & applied to config/settings.yaml. "
              "It takes effect on the next `run`.")
    else:
        print("\nReview the plan above. Re-run with --yes to APPROVE and apply it.")


def cmd_backtest(args):
    from .backtest import backtest, portfolio_backtest

    cfg = load_config()
    if args.strategy:
        strategies = [s for s in cfg.strategies
                      if s.get("name") == args.strategy or s.get("type") == args.strategy]
        if not strategies:
            print(f"No strategy matching '{args.strategy}' in config.")
            return
    else:
        strategies = cfg.enabled_strategies() or cfg.strategies
    if not strategies:
        print("No strategies configured.")
        return
    for sc in strategies:
        try:
            if sc.get("type") in ("haasey", "haasey_v3"):
                # Autonomous multi-stock pickers -> portfolio backtest.
                res = portfolio_backtest(cfg, sc, days=args.days, max_symbols=args.max_symbols)
            else:
                res = backtest(cfg, sc, days=args.days)
            print(res.summary())
        except Exception as exc:  # noqa: BLE001
            print(f"[backtest] {sc.get('name')} failed: {exc}")


def cmd_analyze(args):
    from .reports import analyze_equity, format_report

    stats = analyze_equity(args.path, capital=args.capital)
    if not stats:
        print("No equity CSV found. Run `paperlog` first (during market hours).")
        return
    print(format_report(stats))


def cmd_paperlog(args):
    from .engine.signal_logger import PaperLiveLogger

    cfg = load_config()
    if args.strategy:
        found = False
        for s in cfg.strategies:
            match = s.get("name") == args.strategy or s.get("type") == args.strategy
            s["enabled"] = match
            found = found or match
        if not found:
            print(f"No strategy matching '{args.strategy}' in config.")
            return
    if args.max_symbols:
        for s in cfg.strategies:
            if not s.get("enabled"):
                continue
            uni = s.get("universe")
            if uni == "auto":
                from .data import default_universe
                s["universe"] = default_universe()[: args.max_symbols]
            elif isinstance(uni, list):
                s["universe"] = uni[: args.max_symbols]
    PaperLiveLogger(cfg).run(minutes=args.minutes, poll_seconds=args.poll,
                             respect_session=not args.now)


def cmd_walkforward(args):
    from .backtest import format_folds, walk_forward

    cfg = load_config()
    strat = None
    if args.strategy:
        strat = next((s for s in cfg.strategies
                      if s.get("name") == args.strategy or s.get("type") == args.strategy), None)
    if strat is None:
        strat = next((s for s in cfg.strategies if s.get("type", "").startswith("haasey")), None)
    if strat is None:
        print("No HAASEY-type strategy in config to walk-forward.")
        return
    print(f"Walk-forward: optimize on {args.train_bars} bars, validate on next "
          f"{args.test_bars}, rolling over ~{args.days}d. Backtest = no API cost.")
    folds = walk_forward(cfg, strat, days=args.days, max_symbols=args.max_symbols,
                         train_bars=args.train_bars, test_bars=args.test_bars)
    print(format_folds(folds))


def cmd_supervisor(args):
    from .agent.supervisor import run_supervisor

    result = run_supervisor(load_config(), approve=args.approve)
    print(f"\n[supervisor] {result}")


def cmd_rltest(args):
    from .ml.rl_agent import train_and_evaluate

    cfg = load_config()
    strat = next((s for s in cfg.strategies if s.get("type", "").startswith("haasey")), None)
    if strat is None:
        print("No HAASEY strategy in config.")
        return
    print(f"Training PPO ({args.timesteps} steps/symbol, {args.max_symbols} symbols), "
          f"validating OOS vs baseline. Experimental. No API cost.")
    print(train_and_evaluate(cfg, strat, days=args.days, max_symbols=args.max_symbols,
                             timesteps=args.timesteps).summary())


def cmd_mlsignal(args):
    from .ml.xgb_signal import train_and_evaluate

    cfg = load_config()
    strat = next((s for s in cfg.strategies if s.get("type", "").startswith("haasey")), None)
    if strat is None:
        print("No HAASEY strategy in config.")
        return
    print(f"Training XGBoost signal ({args.days}d, {args.max_symbols} symbols), "
          f"validating out-of-sample vs the rule baseline. No API cost.")
    print(train_and_evaluate(cfg, strat, days=args.days, max_symbols=args.max_symbols).summary())


def cmd_genetic(args):
    from .backtest import genetic_optimize

    cfg = load_config()
    strat = next((s for s in cfg.strategies if s.get("type", "").startswith("haasey")), None)
    if strat is None:
        print("No HAASEY strategy in config to optimize.")
        return
    print(f"Genetic optimize (pop {args.pop}, gens {args.generations}, {args.days}d). "
          f"Walk-forward gated (train->test). No API cost.")
    res = genetic_optimize(cfg, strat, days=args.days, max_symbols=args.max_symbols,
                           pop_size=args.pop, generations=args.generations)
    print(res.summary())


def cmd_optimize(args):
    from .backtest import format_rows, grid_search

    cfg = load_config()
    haasey = next((s for s in cfg.strategies if s.get("type") == "haasey"), None)
    if not haasey:
        print("No HAASEY strategy found in config to optimise.")
        return
    print(f"Grid search over buy / cooldown / trail_atr_mult "
          f"({args.days}d, up to {args.max_symbols or 'all'} symbols). "
          f"Backtest = no API cost.")
    rows = grid_search(cfg, haasey, days=args.days, max_symbols=args.max_symbols)
    print(format_rows(rows, top=args.top))
    if rows:
        best = rows[0]
        print(f"\nBest net P&L: {best.params} -> Rs.{best.net_pnl:,.0f} "
              f"({best.trades} trades, Sharpe {best.sharpe:.2f})")
        print("Reminder: a good backtest is necessary but NOT sufficient. "
              "Prefer a robust combo, not the single highest number (overfitting).")


def cmd_kite_login(args):
    from kiteconnect import KiteConnect

    cfg = load_config()
    s = cfg.secrets
    if not s.kite_api_key:
        print("Set KITE_API_KEY in .env first.")
        return
    kite = KiteConnect(api_key=s.kite_api_key)
    if not args.request_token:
        print("Step 1: open this URL, log in, and copy the `request_token` from the redirect:\n")
        print("   " + kite.login_url())
        print("\nThen run: python -m src.main kite-login --request-token <TOKEN>")
        return
    if not s.kite_api_secret:
        print("Set KITE_API_SECRET in .env to exchange the token.")
        return
    data = kite.generate_session(args.request_token, api_secret=s.kite_api_secret)
    print("\n✅ Paste this into your .env (valid for today):\n")
    print(f"KITE_ACCESS_TOKEN={data['access_token']}")


def cmd_web(args):
    import uvicorn

    uvicorn.run("src.web.app:app", host=args.host, port=args.port, reload=False)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tradingbot", description="Autonomous trading bot (paper-first).")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="run a trading session")
    pr.add_argument("--poll", type=float, default=2.0, help="seconds between ticks")
    pr.add_argument("--sandbox", nargs="?", const=60, default=None,
                    help="ignore market hours; run N seconds (paper testing)")
    pr.set_defaults(func=cmd_run)

    pd_ = sub.add_parser("daemon", help="auto start/stop every trading day")
    pd_.add_argument("--poll", type=float, default=2.0)
    pd_.set_defaults(func=cmd_daemon)

    sub.add_parser("status", help="show P&L, lock state, positions").set_defaults(func=cmd_status)
    sub.add_parser("reset", help="unlock the day (press Run again)").set_defaults(func=cmd_reset)
    sub.add_parser("stop", help="manual kill switch now").set_defaults(func=cmd_stop)

    pc = sub.add_parser("chat", help="natural-language instruction -> proposed config")
    pc.add_argument("instruction")
    pc.add_argument("--yes", action="store_true", help="approve & apply the proposal")
    pc.set_defaults(func=cmd_chat)

    pb = sub.add_parser("backtest", help="backtest enabled strategies")
    pb.add_argument("--days", type=int, default=5)
    pb.add_argument("--strategy", default=None,
                    help="backtest only this strategy (by name or type), even if disabled")
    pb.add_argument("--max-symbols", type=int, default=None,
                    help="HAASEY: cap universe size for a faster run")
    pb.set_defaults(func=cmd_backtest)

    pa = sub.add_parser("analyze", help="summarise a paperlog equity CSV (bot performance)")
    pa.add_argument("path", nargs="?", default=None, help="equity_*.csv path (default: latest)")
    pa.add_argument("--capital", type=float, default=20000.0)
    pa.set_defaults(func=cmd_analyze)

    pp = sub.add_parser("paperlog", help="run LIVE in paper mode; log signals + P&L to CSV")
    pp.add_argument("--minutes", type=float, default=None, help="run for N minutes (default: until market close)")
    pp.add_argument("--poll", type=float, default=60.0, help="seconds between checks")
    pp.add_argument("--strategy", default=None, help="run only this strategy (name or type)")
    pp.add_argument("--max-symbols", type=int, default=None, help="cap HAASEY universe (fewer live fetches)")
    pp.add_argument("--now", action="store_true", help="ignore market hours (test the tool anytime)")
    pp.set_defaults(func=cmd_paperlog)

    pwf = sub.add_parser("walkforward", help="rolling train/test validation (IS vs OOS)")
    pwf.add_argument("--days", type=int, default=60)
    pwf.add_argument("--strategy", default=None)
    pwf.add_argument("--max-symbols", type=int, default=12)
    pwf.add_argument("--train-bars", type=int, default=250)
    pwf.add_argument("--test-bars", type=int, default=125)
    pwf.set_defaults(func=cmd_walkforward)

    psup = sub.add_parser("supervisor", help="regime-aware LangGraph supervisor (proposes tilt, human-gated)")
    psup.add_argument("--approve", action="store_true", help="approve & commit the proposed tilt")
    psup.set_defaults(func=cmd_supervisor)

    prl = sub.add_parser("rltest", help="train+evaluate PPO agent OOS vs baseline (experimental)")
    prl.add_argument("--days", type=int, default=45)
    prl.add_argument("--max-symbols", type=int, default=5)
    prl.add_argument("--timesteps", type=int, default=4000)
    prl.set_defaults(func=cmd_rltest)

    pm = sub.add_parser("mlsignal", help="train+evaluate XGBoost signal OOS vs baseline")
    pm.add_argument("--days", type=int, default=45)
    pm.add_argument("--max-symbols", type=int, default=10)
    pm.set_defaults(func=cmd_mlsignal)

    pg = sub.add_parser("genetic", help="genetic optimizer (walk-forward gated, no API cost)")
    pg.add_argument("--days", type=int, default=45)
    pg.add_argument("--max-symbols", type=int, default=10)
    pg.add_argument("--pop", type=int, default=12)
    pg.add_argument("--generations", type=int, default=6)
    pg.set_defaults(func=cmd_genetic)

    po = sub.add_parser("optimize", help="grid-search HAASEY efficiency params (no API cost)")
    po.add_argument("--days", type=int, default=30)
    po.add_argument("--max-symbols", type=int, default=10)
    po.add_argument("--top", type=int, default=15)
    po.set_defaults(func=cmd_optimize)

    pk = sub.add_parser("kite-login", help="Zerodha daily access-token helper")
    pk.add_argument("--request-token", default=None)
    pk.set_defaults(func=cmd_kite_login)

    pw = sub.add_parser("web", help="launch dashboard + chat")
    pw.add_argument("--host", default="127.0.0.1")
    pw.add_argument("--port", type=int, default=8000)
    pw.set_defaults(func=cmd_web)
    return p


def main(argv=None):
    # Windows consoles default to cp1252 and can't print emoji/₹; force UTF-8.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
