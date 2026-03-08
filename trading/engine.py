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
from core.trading_config import TradingConfig


async def run_market_monitor(bridge, log_func, delay: float | None = None):
    """Run lightweight monitor loop with modular delegation."""
    config = TradingConfig.from_env()
    loop_delay = float(delay) if delay is not None else float(config.loop_delay_seconds)
    min_ev_threshold = float(config.min_ev)
    allocation_fraction = 0.10

    executor = TradeExecutor(risk_config=RiskConfig(ev_threshold=config.min_ev))
    bridge.current_balance = float(executor.get_balance())
    portfolio_manager = PortfolioManager(
        bridge=bridge,
        executor=executor,
        config=config,
    )
    my_hunter = PolymarketScannerHunter(
        bridge=bridge,
        executor=executor,
        config=config,
    )

    while True:
        await asyncio.sleep(loop_delay)

        requested_live = bool(getattr(bridge, "live_trading", False))
        executor.dry_run = not requested_live

        portfolio_manager.manage_portfolio(log_func)

        market, hunter = my_hunter.get_active_markets(log_func)
        if not market or not hunter:
            bridge.status = "❌ No markets found. Waiting..."
            continue

        prepared = my_hunter.prepare_market_signal(market, hunter, log_func)
        if not prepared:
            continue

        signal = prepared["signal"]
        token_id = prepared["token_id"]
        asset_type = prepared["asset_type"]
        question = prepared["question"]
        model_used = prepared["model_used"]
        price_yes = float(prepared["poly_price"])

        my_hunter.mark_seen(token_id)

        ev_yes = (float(signal.fair_value) / float(price_yes) - 1.0) if price_yes > 0 else -1.0
        price_no = max(1e-9, 1.0 - float(price_yes))
        fair_no = 1.0 - float(signal.fair_value)
        ev_no = (fair_no / price_no - 1.0) if price_no > 0 else -1.0

        best_side = "YES" if ev_yes > ev_no else "NO"
        final_ev = max(ev_yes, ev_no)

        diagnostic_msg = f"[EV-MATH] YES(P: {price_yes:.3f}, FV: {float(signal.fair_value):.3f}, EV: {ev_yes:.2f}) | NO(P: {price_no:.3f}, FV: {fair_no:.3f}, EV: {ev_no:.2f}) | PICK: {best_side}"
        print(diagnostic_msg)
        bridge.terminal_logs.appendleft(diagnostic_msg)

        if float(final_ev) < float(min_ev_threshold):
            log_func(
                "REJECTED",
                asset_type,
                token_id,
                {
                    "market_name": question,
                    "reason": "EV below dynamic threshold",
                    "ev_yes": round(float(ev_yes), 4),
                    "ev_no": round(float(ev_no), 4),
                    "side": best_side,
                    "ev": round(float(final_ev), 4),
                    "threshold": min_ev_threshold,
                },
            )
            my_hunter.mark_seen(token_id)
            continue

        total_equity = float(bridge.current_balance) + float(bridge.open_position_value)
        bet_amount = total_equity * float(allocation_fraction)

        if float(bridge.current_balance) < float(bet_amount):
            capital_ready = portfolio_manager.free_up_capital(float(bet_amount), log_func)
            if not capital_ready:
                log_func(
                    "REJECTED",
                    asset_type,
                    token_id,
                    {
                        "market_name": question,
                        "reason": "Insufficient capital after liquidation",
                        "required_amount": round(float(bet_amount), 4),
                        "current_balance": round(float(bridge.current_balance), 4),
                    },
                )
                my_hunter.mark_seen(token_id)
                continue

        executed = executor.evaluate_and_execute(
            market=prepared["market"],
            fair_value=float(signal.fair_value),
            ev=float(final_ev),
            current_poly_price=price_yes,
            bet_amount_usd=float(bet_amount),
            side=best_side,
            log_func=log_func,
        )

        if not executed:
            my_hunter.mark_seen(token_id)

        if executed:
            bridge.current_balance = max(0.0, float(bridge.current_balance) - float(bet_amount))

        log_func(
            "TRACK",
            asset_type,
            token_id,
            {
                "market_name": question,
                "model_used": model_used,
                "fair": round(float(signal.fair_value), 4),
                "ev": round(float(final_ev), 4),
                "ev_yes": round(float(ev_yes), 4),
                "ev_no": round(float(ev_no), 4),
                "side": best_side,
                "kelly": round(float(signal.kelly_size), 4),
                "bet_usd": round(float(bet_amount), 2),
                "executed": bool(executed),
                "total_equity": round(float(total_equity), 4),
                "allocation_fraction": allocation_fraction,
            },
        )
