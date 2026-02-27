"""
Engine: Thin orchestration layer.

Main loop delegates responsibilities to:
- PortfolioManager (risk_manager.py): position exits (TP/SL/EV convergence)
- PolymarketScannerHunter (hunters/polymarket_scanner.py): one full market scan pass
"""

import asyncio
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from trading.executor import TradeExecutor, RiskConfig
from hunters import PolymarketScannerHunter
from trading.risk_manager import PortfolioManager
from trading.budget_manager import BudgetManager
from core.trading_config import TradingConfig


async def run_market_monitor(bridge, log_func, delay: float | None = None):
    """Run lightweight monitor loop with modular delegation."""
    config = TradingConfig.from_env()
    loop_delay = float(delay) if delay is not None else float(config.loop_delay_seconds)

    executor = TradeExecutor(risk_config=RiskConfig(ev_threshold=config.min_ev))
    budget_manager = BudgetManager(
        bridge=bridge,
        config=config,
        initial_balance=executor.get_balance(),
    )
    portfolio_manager = PortfolioManager(
        bridge=bridge,
        executor=executor,
        config=config,
    )
    my_hunter = PolymarketScannerHunter(
        bridge=bridge,
        executor=executor,
        budget_manager=budget_manager,
        config=config,
    )

    while True:
        await asyncio.sleep(loop_delay)
        portfolio_manager.manage_portfolio(log_func)
        my_hunter.scan_markets(log_func)
