
# PolyBot — Quantitative Arbitrage Terminal for Polymarket

A fully automated trading bot that hunts [Polymarket](https://polymarket.com) prediction markets for positive expected-value opportunities, evaluates them with domain-specific pricing models, and executes risk-managed trades — all surfaced through a live Streamlit dashboard.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Running the Dashboard](#running-the-dashboard)
- [Trading Modes](#trading-modes)
- [Running Tests](#running-tests)
- [Deployment](#deployment)
- [Environment Variables Reference](#environment-variables-reference)

---

## Overview

PolyBot scans Polymarket's prediction markets for mispriced contracts using real-world reference data (crypto spot prices, weather forecasts, macroeconomic indicators). When it finds a market where the on-chain price diverges from its fair-value estimate by more than the configured EV threshold, it enters a position — subject to daily budget limits, bankroll controls, and take-profit/stop-loss guards.

**Key capabilities:**

- Market discovery across crypto, weather, and macro asset classes
- Domain-specific fair-value models (Black-Scholes/Heston for crypto, normal-distribution for weather/economy)
- Kelly-criterion-based position sizing with hard budget caps
- Dry-run, paper-trading, and live-trading modes
- Real-time dashboard with equity curve, open positions, and EV distribution
- Docker-based deployment to AWS EC2 via GitHub Actions CI/CD

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Dashboard                      │
│              ui/dashboard.py  ·  ui/components.py           │
└─────────────────────────┬───────────────────────────────────┘
                          │ shared state (DataBridge singleton)
┌─────────────────────────▼───────────────────────────────────┐
│               SequentialTradingPipeline                      │
│                 trading/decision_pipeline.py                 │
│                                                             │
│   ┌────────────┐   ┌─────────────┐   ┌──────────────────┐  │
│   │  Hunters   │──▶│   Brains    │──▶│  TradeExecutor   │  │
│   │ (discover) │   │ (price/EV)  │   │ (submit orders)  │  │
│   └────────────┘   └─────────────┘   └──────────────────┘  │
│                                              │               │
│                         ┌────────────────────┘               │
│                         │   PortfolioManager / BudgetMgr     │
│                         │   (risk, P&L, spend limits)        │
│                         └─────────────────────────────────── │
└─────────────────────────────────────────────────────────────┘
                          │
                   SQLite (trades.db)
```

**Core patterns used:**

| Pattern | Where |
|---|---|
| Template Method | `BaseBrain.evaluate()` orchestrates; subclasses override `_calculate_probability()` |
| Strategy | `BaseHunter` interface; `CryptoHunter`, `WeatherHunter`, `EconomyHunter` |
| Singleton | `DataBridge` — Streamlit-cached shared state between dashboard and engine |
| Sequential Pipeline | Hunt → Evaluate → Risk-check → Execute in `SequentialTradingPipeline` |
| Factory | `get_brain_for_asset_type()` returns the right pricing model by asset class |

---

## Project Structure

```
PolyBot/
│
├── brains/                    # Fair-value pricing models
│   ├── base.py                # BaseBrain: EV calc, Kelly sizing, tradability
│   ├── crypto.py              # HybridCryptoBrain: Black-Scholes + Heston
│   ├── weather.py             # WeatherBrain: normal distribution on temperature
│   └── economy.py            # EconomyBrain: normal distribution on macro indicators
│
├── core/                      # Shared infrastructure
│   ├── models.py              # MarketData, TradeSignal, Position dataclasses
│   ├── trading_config.py      # TradingConfig loaded from config/.env
│   └── bridge.py              # DataBridge singleton (dashboard ↔ engine state)
│
├── hunters/                   # Market discovery and reference-data fetching
│   ├── base.py                # BaseHunter interface + Polymarket pagination logic
│   ├── crypto.py              # CryptoHunter (Binance anchor prices)
│   ├── weather.py             # WeatherHunter (OpenWeather API)
│   ├── economy.py             # EconomyHunter (FRED API)
│   ├── parsers.py             # Strike extraction utilities
│   └── clients/               # API clients (Binance, FRED)
│
├── trading/                   # Execution pipeline and risk management
│   ├── decision_pipeline.py   # SequentialTradingPipeline + run_market_monitor()
│   ├── executor.py            # TradeExecutor: dry/paper/live order submission
│   ├── budget_manager.py      # Daily spend limits and bankroll enforcement
│   └── risk_manager.py        # PortfolioManager: take-profit, stop-loss, fair-value lookup
│
├── ui/                        # Streamlit dashboard
│   ├── dashboard.py           # Main app: engine startup, layout, live refresh
│   ├── data_manager.py        # SQLite schema, trade/event logging, stats queries
│   └── components.py          # Equity curve, EV chart, positions table
│
├── tests/
│   ├── unit/                  # Per-module unit tests
│   ├── integration/           # Multi-component integration tests
│   ├── e2e/                   # Full dry-run pipeline end-to-end tests
│   └── conftest.py            # Pytest fixtures
│
├── config/
│   ├── .env                   # Deployment credentials and runtime settings
│   ├── requirements.txt       # Python dependencies
│   └── Docker/
│       ├── Dockerfile
│       └── docker-compose.yml
│
├── .github/
│   └── workflows/
│       ├── deploy.yml         # Build → ECR push → EC2 deploy on push to main
│       └── claude.yml         # Claude Code bot for PR/issue automation
│
├── polymarket.py              # PolymarketClient (Gamma API + CLOB balance)
└── pytest.ini
```

---

## How It Works

### 1. Market Discovery (Hunters)

Each hunter queries the Polymarket Gamma API for open markets matching its asset class. It then fetches a real-world **anchor value** from an external source:

| Hunter | Markets | Anchor Source |
|---|---|---|
| `CryptoHunter` | BTC/ETH/SOL price markets | Binance spot price |
| `WeatherHunter` | Temperature prediction markets | OpenWeather API |
| `EconomyHunter` | Fed Rate, CPI, GDP markets | FRED API |

### 2. Fair-Value Pricing (Brains)

Each discovered market is passed to the matching `Brain`, which calculates the probability the market resolves YES based on the anchor value and a statistical model:

- **Crypto**: Black-Scholes for short TTE, Heston model for longer-dated contracts. Uses per-asset implied volatility (BTC 50%, ETH 70%, SOL 90%).
- **Weather**: Normal distribution around the forecast with a configurable standard deviation.
- **Economy**: Normal distribution around the current macro reading with historical volatility.

The brain returns an **expected value (EV)** = `(fair_prob - market_price)`. A positive EV means the market is underpriced; negative means overpriced.

### 3. Trade Decision

The pipeline filters markets through several gates before executing:

1. EV must exceed `MIN_EV` (default 0.30)
2. Time to expiry must be between `MIN_TTE_MINUTES` and `MAX_TTE_DAYS`
3. Daily spend must be below `DAILY_LIMIT_USD`
4. Available balance must exceed `MIN_TRADING_BALANCE`
5. Position size is Kelly-criterion-scaled, capped at `MAX_BET_SIZE_USD`

### 4. Position Management

Open positions are monitored each cycle. A position is closed when:
- PnL reaches `TAKE_PROFIT_PCT` (default +20%)
- PnL falls below `STOP_LOSS_PCT` (default -50%)
- EV drops below `MIN_HOLD_EV` (default -0.10) on re-evaluation

### 5. Dashboard

The dashboard runs the trading engine in a background thread and auto-refreshes every 2 seconds via `st.fragment`. Three views:

- **Hunter** — live scan history table with EV coloring + terminal log feed
- **Portfolio** — open positions and EV distribution chart
- **Balance** — equity curve, win rate, and trade stats

---

## Getting Started

### Prerequisites

- Python 3.11+
- A Polymarket account with a funded proxy wallet (for live trading)
- API keys for OpenWeather and/or FRED (for weather/macro markets)

### Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/your-org/polybot.git
cd polybot

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r config/requirements.txt

# 4. Configure credentials
cp config/.env config/.env.local   # or edit config/.env directly
# Fill in POLYMARKET_PRIVATE_KEY and POLYMARKET_PROXY_ADDRESS
```

---

## Configuration

All configuration lives in `config/.env`. Copy the template and fill in your values:

```dotenv
# Polymarket wallet credentials (required)
POLYMARKET_PRIVATE_KEY=0x...
POLYMARKET_PROXY_ADDRESS=0x...
SIGNATURE_TYPE=2

# Trading mode (safe defaults — see Trading Modes section)
DRY_RUN=True
PAPER_TRADE_MODE=False

# Risk parameters
MIN_EV=0.30
DAILY_LIMIT_USD=5.0
MAX_BET_SIZE_USD=3.0
BANKROLL_USD=1000.0
MIN_TRADING_BALANCE=5.0

# Position exit thresholds
TAKE_PROFIT_PCT=0.20
STOP_LOSS_PCT=-0.50
MIN_HOLD_EV=-0.10

# Market filter
MIN_TTE_MINUTES=60
MAX_TTE_DAYS=180

# External API keys (optional — enables weather/macro hunters)
OPENWEATHER_API_KEY=...
FRED_API_KEY=...
```

---

## Running the Dashboard

```bash
# From the project root
streamlit run ui/dashboard.py
```

Opens at **http://localhost:8501**.

The dashboard will attempt to start the trading engine immediately on load. Ensure `config/.env` contains valid credentials before launching, or the app will show a startup error.

---

## Trading Modes

The bot supports three trading modes controlled by environment variables:

| Mode | `DRY_RUN` | `PAPER_TRADE_MODE` | Behaviour |
|---|---|---|---|
| **Dry Run** | `True` | `False` | Scans and evaluates markets, logs decisions, submits no orders |
| **Paper Trade** | `False` | `True` | Simulates trades using a virtual balance (`PAPER_BALANCE_USD`) |
| **Live** | `False` | `False` | Submits real orders to Polymarket CLOB |

The sidebar toggle in the dashboard also switches between Dry Run and Live at runtime without a restart.

> Start with `DRY_RUN=True` to observe the engine's decisions before committing real funds.

---

## Running Tests

```bash
# Run all tests
pytest

# Run only unit tests
pytest tests/unit/

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/unit/test_brains_crypto.py
```

Test configuration is in `pytest.ini`. Async tests are handled automatically via `pytest-asyncio`.

---

## Deployment

The project ships with a Dockerfile and a GitHub Actions workflow for one-command deployment to AWS EC2.

### Docker (local)

```bash
docker compose -f config/Docker/docker-compose.yml up --build
```

Mounts `config/.env` for credentials and persists `trades.db` via a volume.

### GitHub Actions (CI/CD)

Pushing to `main` triggers `.github/workflows/deploy.yml`:

1. Builds the Docker image
2. Pushes to Amazon ECR (`eu-north-1`)
3. Starts the EC2 instance if stopped
4. Deploys via AWS SSM (`docker compose pull && up -d`)

Required GitHub secrets: `AWS_ROLE_ARN`, ECR repository URL, EC2 instance ID.

---

## Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `POLYMARKET_PRIVATE_KEY` | — | Private key for CLOB order signing |
| `POLYMARKET_PROXY_ADDRESS` | — | Proxy wallet address |
| `POLYGON_PRIVATE_KEY` | — | Alias for `POLYMARKET_PRIVATE_KEY` |
| `POLY_ADDRESS` | — | Alias for `POLYMARKET_PROXY_ADDRESS` |
| `SIGNATURE_TYPE` | `2` | Polymarket signature scheme (2 = proxy/funder) |
| `DRY_RUN` | `True` | Disable order submission |
| `PAPER_TRADE_MODE` | `False` | Simulate trades with virtual balance |
| `PAPER_BALANCE_USD` | `1000.0` | Starting virtual balance for paper trading |
| `MIN_EV` | `0.30` | Minimum expected value to enter a trade |
| `MIN_TTE_MINUTES` | `60` | Minimum time-to-expiry in minutes |
| `MAX_TTE_DAYS` | `180` | Maximum time-to-expiry in days |
| `DAILY_LIMIT_USD` | `5.0` | Maximum USD to spend per day |
| `MAX_BET_SIZE_USD` | `3.0` | Maximum single trade size in USD |
| `BANKROLL_USD` | `1000.0` | Total bankroll for Kelly sizing |
| `MIN_TRADING_BALANCE` | `5.0` | Minimum wallet balance to allow new trades |
| `TAKE_PROFIT_PCT` | `0.20` | Close position at +20% PnL |
| `STOP_LOSS_PCT` | `-0.50` | Close position at -50% PnL |
| `MIN_HOLD_EV` | `-0.10` | Close position if re-evaluated EV drops below this |
| `ENGINE_LOOP_DELAY` | `2.0` | Seconds between scan cycles |
| `TRADES_DB_PATH` | `/app/trades.db` | SQLite database path |
| `OPENWEATHER_API_KEY` | — | Required for WeatherHunter |
| `FRED_API_KEY` | — | Required for EconomyHunter |
