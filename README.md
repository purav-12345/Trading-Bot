# Autonomous Indian Equities Trading System

HAASEY is a personal-use, paper-first automated trading framework specifically designed for Indian equities. It operates by scanning a predefined universe of stocks, scoring them using a combination of fundamental, technical, and news sentiment factors, and adapting its behavior to the current market regime. The system sizes positions dynamically based on risk parameters and enforces a strict daily profit/loss kill-switch. The foundational philosophy of this project is to validate strategies through rigorous walk-forward testing, ensuring that developers are not fooled by in-sample backtesting fantasies.

## ⚠️ Honest Disclaimer: Read Before Proceeding

This bot does not generate a guaranteed profit, and it is crucial to acknowledge that no bot does. 
* **Zero Profitable Edge:** Every rigorous out-of-sample walk-forward test conducted (0 out of 9 windows positive) demonstrates that there is no profitable edge on liquid NSE large-cap stocks once trading costs are factored in.
* **Retail Trading Reality:** According to the Securities and Exchange Board of India (SEBI), 93% of retail Futures and Options (F&O) traders lose money. 
* **Project Purpose:** The true value of this project lies in enforcing discipline and providing honest algorithmic validation, rather than generating returns. 
* **Paper Trading Only:** The system runs in paper mode (simulated fills) by default. You must not risk real money without securing a long, positive, out-of-sample track record in live-paper trading.
<img width="2752" height="1536" alt="Gemini_Generated_Image_hwsk1hhwsk1hhwsk" src="https://github.com/user-attachments/assets/207d8d2a-58d5-42bf-b636-047db3deb86e" />
<img width="2752" height="1536" alt="Gemini_Generated_Image_uhag72uhag72uhag" src="https://github.com/user-attachments/assets/816150e3-bebb-4ca8-a6ab-cdc424e7e54d" />



## Core Framework Features

This project provides an end-to-end research and execution architecture:

* **Autonomous Stock Picker (HAASEY):** A regime-aware, multi-factor scoring engine.
* **Comprehensive Risk Engine:** Enforces a daily profit/loss kill-switch, caps risk per trade, and manages auto square-offs.
* **Honest Validation Suite:** Includes a robust cost model and walk-forward testing (comparing in-sample versus out-of-sample performance).
* **Separated AI Layer:** Utilizes FinBERT for sentiment analysis, Gemini for chat/parsing, and LangGraph for a human-approval agent. The core design rule dictates that LLMs and ML models never execute orders directly; they only propose actions requiring human approval, keeping the actual trade loop restricted to pure, deterministic Python.
* **Gated ML Experiments:** Incorporates experimental features like XGBoost signals with SHAP, a genetic optimizer, and PPO/RL models. These are retained only if they outperform an out-of-sample baseline (notably, none have succeeded thus far).

## The Trading Cycle (How It Works)

A single operational cycle follows a strict six-step pipeline:

1. **REGIME:** The system classifies the broader market using NIFTY and VIX data into BULL, BEAR, or CHOPPY states.
2. **SCAN:** The system evaluates each stock in the target universe:
   * Resamples price data to 15-minute decision bars.
   * *If holding a position:* Runs EXIT checks (take-profit, ATR trailing stop, or square-off limits).
   * *If flat (no position):* Evaluates entry criteria including TREND (price > SMA50 and > EMA20), 60-minute higher-timeframe confluence, and time-window validity. It then calculates a 7-factor score; if this score meets the buy threshold, it becomes a candidate.
3. **RANK:** Candidates are sorted by their generated scores, and the system targets the best options it can afford based on available capital.
4. **SIZE:** Position sizing is determined by multiplying ATR risk by a regime multiplier and a fractional-Kelly criterion.
5. **GATE:** Every generated order passes through the Risk Engine, which can resize or reject the order before it reaches the broker.
6. **GUARD:** At every tick, the system monitors the daily profit/loss kill-switch, flattening positions and locking the system if limits are breached.

## Technology Stack

| Layer | Technology |
| :--- | :--- |
| **Language** | Python 3.11 |
| **Broker (Live)** | Zerodha Kite Connect (`kiteconnect`) |
| **Market Data** | `yfinance` (Real NSE prices, candles, fundamentals) |
| **Data & Math** | `pandas`, `numpy` |
| **Scheduling** | `APScheduler`, `pytz` (IST) |
| **AI (Sentiment)** | FinBERT (`transformers` + `torch`, ProsusAI/finbert) — Local, free |
| **AI (Chat/Agent)** | Gemini (`google-genai`, `gemini-2.5-flash-lite`), LangGraph |
| **ML Experiments** | `xgboost`, `scikit-learn`, `shap`, `stable-baselines3`, `gymnasium` |
| **News Feed** | Google News RSS (Free) |
| **Web UI** | FastAPI + `uvicorn` |
| **Storage** | SQLite (Orders, daily P&L, audit logs) |
| **Testing** | `pytest` (17 tests) |

## Strategy Parameters: `haasey_v3`

The flagship strategy is calibrated for a ₹20,000 account. The current tuning configuration is designed to filter out market noise and limit churn:

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Capital** | ₹20,000 | Baseline account size. |
| **Universe** | ~45 stocks | Autoscan liquid NSE names to select the best candidates. |
| **Decision Bar** | 15 minutes | Trades on 15-minute bars to avoid tick-level noise. |
| **Thresholds** | 0.46 / 0.65 | Buy and Strong Buy score thresholds (0.46 casts a wider net). |
| **Take Profit** | 1.0% | Targets a meaningful 15-minute move to prevent over-trading/churn. |
| **Trail ATR Mult** | 2.0 | Employs a tighter trailing stop for faster loss mitigation. |
| **Cooldown** | 60 minutes | Forces holding for ≥4 bars before a stop exit to reduce whipsaws. |
| **Profit/Cost Ratio**| 2.0 | Only executes if the expected move is greater than 2× the round-trip transaction costs. |
| **Trade Windows** | 09:30–11:30, 13:30–15:00 | Skips the historically choppy midday market lull. |
| **Trend Filters** | 50 SMA / 20 EMA | Restricts buying to established uptrends. |
| **HTF Filters** | 60 min / 20 SMA | Requires the 60-minute higher timeframe trend to agree (confluence). |
| **Regime** | 50 SMA / 20.0 VIX | Calculates regime based on NIFTY SMA50 and the India VIX high. |
| **Regime Weights** | True | BULL regimes use momentum-led weights; BEAR regimes pivot to value/quality-led weights (Live). |
| **Per-Trade Risk** | 0.015 | Risks approximately 1.5% of total capital per trade. |
| **Max Position** | 0.12 | Allocates a maximum of 12% of total capital to any single stock. |
| **Max Positions** | 25 | Uncapped in theory, but available cash practically limits this to ~5–8 names. |

**7-Factor Scoring:** The live strategy applies weights across Value (0.25), Quality (0.22), Growth (0.22), Momentum (0.13), Technical (0.08), Sentiment (0.05), and Stability (0.05). In backtests, weights are flipped to favor technical/momentum factors, as testing fundamental data introduces look-ahead bias. Regime-aware logic overrides these weights in live execution.

**Risk Limits:** The system halts trading for the day if it hits a daily profit target of ₹30,000 or a daily maximum loss of ₹2,000.

## Validation & Results

All reported results are honest, paper-based, and factor in modeled costs including Zerodha brokerage, STT, and estimated slippage. 

| Test Type | Result | Takeaway |
| :--- | :--- | :--- |
| **Walk-Forward** | 0/9 folds positive | Rigorous out-of-sample (OOS) testing shows no edge. |
| **Tuned Walk-Forward**| 0/9 folds positive | OOS loss halved (−₹422 → −₹239), but still negative. |
| **XGBoost (OOS)** | −₹58 vs baseline −₹198 | The ML signal lost less money but was not profitable. |
| **Genetic Optimizer** | IS +₹275 → OOS +₹14 | Severe overfitting shrinkage from in-sample to out-of-sample. |
| **PPO / RL** | −₹515 vs baseline −₹87 | Underperformed the baseline significantly; dropped from active use. |
| **Live Paper (1 day)** | +₹25 (+0.13%) | Statistically insignificant noise on a rising market day. |

**Conclusion on Results:** The framework is highly sophisticated and fully operational, but it is not a proven money-maker. It successfully fulfills its primary objective: honestly proving the absence of a tradable edge under current parameters.

## 📂 Project Architecture & Directory Layout

The codebase enforces a strict separation between configuration, deterministic execution, data pipelines, risk management, and offline AI/ML advisory experiments:
<img width="2752" height="1536" alt="Gemini_Generated_Image_3fm6tp3fm6tp3fm6" src="https://github.com/user-attachments/assets/4ca6af03-a7a4-44a3-8c38-8451510d14e2" />

```text
haasey-bot/
├── config/
│   └── settings.yaml       # Core configuration: session hours, risk parameters, fees, and active strategy selection
├── src/
│   ├── strategies/         # Strategy implementations: flagship haasey_v3, orb_vwap, supertrend_vwap, rsi_reversion
│   ├── data/               # Market data pipelines: NIFTY/VIX regime detection, universe scanner, fundamentals
│   ├── ai/                 # AI & NLP modules: local FinBERT sentiment scoring, Gemini interface, Google News RSS
│   ├── agent/              # LangGraph multi-step workflows & human-in-the-loop regime supervisor
│   ├── risk/               # Real-time risk engine: ATR/Kelly dynamic position sizing, daily kill-switch
│   ├── engine/             # Core execution loop: APScheduler, state machines, paper logging, Zerodha Kite gateway
│   ├── backtest/           # Validation suite: walk-forward testing engine, portfolio simulator, cost models
│   ├── ml/                 # Gated ML experiments: XGBoost signal models, SHAP explanations, PPO/RL environments
│   └── web/
│       └── app.py          # FastAPI web dashboard, real-time telemetry, and interactive supervisor chat
└── tests/                  # 17 pytest suites covering risk calculations, factor scoring, and order execution
```

## Installation & Execution

To set up the environment and run the system locally:

```bash
# Set up virtual environment
python -m venv .venv

# Activate environment (Windows)
.venv\Scripts\activate            
# OR activate environment (Linux/Mac)
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt




# Configure environment variables (paper trading requires no keys)
cp .env.example .env              
# Add GEMINI_API_KEY to .env if using the AI supervisor layer
