# 🤖 PolyBot: Quantitative Prediction Market Arbitrageur

PolyBot is a modular, high-performance trading framework designed to identify, evaluate, and execute trades on Polymarket. By combining real-time data from global anchors (Binance, FRED) with quantitative probability models, PolyBot automates the search for positive Expected Value (+EV) opportunities.

## 🌟 Key Features

- **Multi-Asset Hunting:** Specialized "Hunters" for Crypto (Price Action), Economy (Interest Rates/CPI), and Weather.
- **Quantitative "Brains":** Real-time probability estimation using Black-Scholes models and Normal Distribution curves.
- **Smart Risk Management:** Automated position sizing using the Kelly Criterion to maximize compound growth while protecting the bankroll.
- **Precision Deep Scanning:** Advanced text-parsing engine that concatenates group market headers with sub-market outcomes to extract strikes accurately.
- **Memory & Cooldowns:** Built-in 10-minute cooldown cache to prevent the bot from getting "stuck" on low-EV markets.
- **Risk Ceilings:** Global daily USD spending limits and per-trade "Half-Kelly" safety clamps.

## 🏗️ Architecture & Techniques

The project follows a strict Separation of Concerns (SoC) architecture, making it highly modular and easy to extend:

- **Network Layer (`clients/`):** Isolated API clients for Polymarket, Binance, and the Federal Reserve (FRED).
- **Parsing Layer (`parsers/`):** Specialized regex engines that translate messy market strings into clean numeric strikes.
- **Scouting Layer (`hunters/`):** Template-method hunters + scanner coordination across market types.
- **Quant Layer (`brains/`):** Statistical engines that calculate fair value vs market price.
- **Core Domain (`core/`):** Shared models, runtime bridge, and trading configuration.
- **Trading Layer (`trading/`):** Execution, budget controls, risk manager, and decision pipeline handlers.
- **Orchestration Layer (`orchestration/engine.py`):** Thin monitor loop wiring scanner + portfolio manager.
- **UI Layer (`ui/`):** Streamlit dashboard, UI components, and history/data management.

### Current Project Tree (high level)

```text
PolyBot/
├─ brains/
├─ clients/
├─ core/
├─ hunters/
├─ orchestration/
├─ parsers/
├─ trading/
├─ ui/
├─ dashboard.py            # compatibility launcher
└─ requirements.txt
```

## 🧰 Tech Stack

- **Language:** Python 3.10+
- **Networking:** `curl_cffi` (for impersonating browser TLS fingerprints to bypass API blocks)
- **Math:** NumPy, SciPy (for cumulative distribution functions)
- **Formatting:** LaTeX-style math for financial modeling

## 🚀 Getting Started

### 1) Prerequisites

- Python 3.10 or higher
- A Polymarket account (API keys required for execution)
- Optional: FRED API key for economic data

### 2) Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/PolyBot.git
cd PolyBot

# Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3) Configuration

Create a `.env` file in the root directory:

```env
POLYMARKET_API_KEY=your_key
BINANCE_API_KEY=your_key
FRED_API_KEY=your_key
DAILY_LIMIT_USD=100.0
MAX_RISK_PER_TRADE=0.05
```

## 🛠️ Manual & Activation

### Starting the Live Bot

To launch the full engine with the real-time dashboard:

```bash
streamlit run ui/dashboard.py
```

Backward-compatible launcher (also supported):

```bash
streamlit run dashboard.py
```

### Developer Mode (Debugging)

To see the bot's internal thought process (regex matches, EV math, and filter rejections), ensure `DEBUG=True` is set in your environment. This outputs logs like:

```text
[DEBUG] Evaluating: Bitcoin above 68,000 | Price: 0.63 | Fair: 0.44 | EV: -29%
[ENGINE] Market has low EV. Entering 10m cooldown.
```
