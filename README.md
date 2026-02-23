# 🛰️ PolyBot: Quantitative Arbitrage Terminal

A high-performance, asynchronous trading dashboard designed to hunt and visualize probability-based arbitrage opportunities between the **Binance Spot Market** and the **Polymarket Central Limit Order Book (CLOB)**.

![PolyBot Terminal](https://img.shields.io/badge/Status-Active-brightgreen) ![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)

## 🧠 How it Works

PolyBot operates on the premise of **latency exploitation and probability pricing**. 
1. **The Engine (Hunter):** Bypasses standard API rate limits to directly scan the Polymarket CLOB for active, high-volume Bitcoin prediction markets (e.g., *"Will BTC reach $100k?"*).
2. **The Brain:** Pulls the live global truth (Binance Spot BTC price) and calculates the mathematical Fair Value of the Polymarket contract using Cumulative Distribution Functions (CDF).
3. **The Edge:** Continuously compares the Polymarket Orderbook Midpoint against the Math-Fair Value. If Polymarket lags behind a sudden Binance price movement, the Expected Value (EV) spikes, signaling a trading edge.

## 🚀 Key Features

* **Auto-Discovery:** Automatically parses stringified JSON from Polymarket's Gamma/Events API to find active, liquid markets without manual Token ID configuration.
* **Self-Healing Logic:** Automatically blacklists dead, expired, or halted orderbooks and hunts for the next available market.
* **Asynchronous Processing:** The data engine runs on a background `asyncio` thread, allowing the UI to remain highly responsive.
* **Execution Ledger:** A local SQLite database (`trades.db`) logs all simulated paper trades seamlessly.
* **Quant-Style UI:** Built with Streamlit, utilizing native metrics, delta color-coding, and layout containers for a professional trading desk feel.

## 🛠️ Installation & Setup

**1. Clone the repository:**
```bash
git clone [https://github.com/YOUR-USERNAME/polybot.git](https://github.com/YOUR-USERNAME/polybot.git)
cd polybot
