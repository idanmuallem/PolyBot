"""
Engine: Thin orchestration layer.

Main loop delegates responsibilities to:
- PortfolioManager (risk_manager.py): position exits (TP/SL/EV convergence)
- PolymarketScannerHunter (hunters/polymarket_scanner.py): one full market scan pass
"""

import asyncio
from datetime import datetime, timezone
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
    max_bet_size_usd = float(config.max_bet_size_usd)
    min_trade_usd = 5.0

    executor = TradeExecutor(risk_config=RiskConfig(ev_threshold=config.min_ev))
    restored_balance = float(getattr(bridge, "current_balance", 0.0) or 0.0)
    if restored_balance > 0.0:
        bridge.current_balance = restored_balance
    else:
        bridge.current_balance = float(executor.get_balance())

    if float(getattr(bridge, "starting_balance", 0.0) or 0.0) <= 0.0:
        bridge.starting_balance = float(bridge.current_balance)
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

    bridge.spent_today = float(getattr(bridge, "spent_today", 0.0) or 0.0)
    spent_today = float(bridge.spent_today)
    spend_day = datetime.now(timezone.utc).date()
    start_of_day_equity = float(getattr(bridge, "start_of_day_equity", 0.0) or 0.0)

    while True:
        await asyncio.sleep(loop_delay)

        current_day = datetime.now(timezone.utc).date()
        if current_day != spend_day:
            spent_today = 0.0
            spend_day = current_day
            start_of_day_equity = 0.0
            bridge.spent_today = 0.0
            bridge.daily_spend = 0.0
            bridge.start_of_day_equity = 0.0

        requested_live = bool(getattr(bridge, "live_trading", False))
        executor.dry_run = not requested_live

        portfolio_manager.manage_portfolio(log_func)

        cash_balance = float(bridge.current_balance)
        open_positions_value = float(bridge.open_position_value)
        total_equity = float(cash_balance) + float(open_positions_value)

        if float(start_of_day_equity) <= 0.0:
            start_of_day_equity = float(total_equity)
            bridge.start_of_day_equity = float(start_of_day_equity)

        daily_cap = min(float(config.daily_limit_usd), max(0.0, float(start_of_day_equity)))
        allowed_remaining = max(0.0, float(daily_cap) - float(spent_today))
        bridge.daily_spend = float(spent_today)

        log_func(
            "LOOP-SUMMARY",
            "Engine",
            f"day:{spend_day.isoformat()}",
            {
                "cash": round(float(cash_balance), 4),
                "spent_today": round(float(spent_today), 4),
                "daily_cap": round(float(daily_cap), 4),
                "allowed_remaining": round(float(allowed_remaining), 4),
                "open_positions_value": round(float(open_positions_value), 4),
                "start_of_day_equity": round(float(start_of_day_equity), 4),
            },
        )

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

        target_bet_unclamped = float(total_equity) * float(allocation_fraction)
        target_bet = min(float(target_bet_unclamped), float(max_bet_size_usd))
        available_cash = float(cash_balance)
        freed_cash = 0.0

        if float(available_cash) < float(target_bet):
            print(
                f"[ENGINE] Insufficient cash (${available_cash:.2f}) for target bet (${target_bet:.2f}). "
                "Triggering portfolio optimization..."
            )
            try:
                freed_cash = float(
                    portfolio_manager.optimize_for_candidate(
                        float(final_ev),
                        min_improvement=0.10,
                        log_func=log_func,
                    )
                )
            except Exception as exc:
                print(f"[ENGINE] Portfolio optimization failed: {exc}")
                freed_cash = 0.0
            available_cash = float(available_cash) + float(freed_cash)
            bridge.current_balance = float(available_cash)

        planned_bet = min(float(target_bet), float(allowed_remaining))

        if float(allowed_remaining) < float(min_trade_usd):
            reason_msg = f"REJECTED: daily_limit_reached (Spent: ${float(spent_today):.2f} / Cap: ${float(daily_cap):.2f})"
            print(f"[REJECTED] {reason_msg}")
            log_func(
                "REJECTED",
                asset_type,
                token_id,
                {
                    "market_name": question,
                    "reason": "daily_limit_reached",
                    "message": reason_msg,
                    "spent_today": round(float(spent_today), 4),
                    "daily_cap": round(float(daily_cap), 4),
                    "allowed_remaining": round(float(allowed_remaining), 4),
                    "target_bet": round(float(target_bet), 4),
                    "max_bet_size_usd": round(float(max_bet_size_usd), 4),
                },
            )
            my_hunter.mark_seen(token_id)
            continue

        bet_amount = min(float(planned_bet), float(available_cash))

        if float(bet_amount) < float(min_trade_usd):
            needed = max(float(min_trade_usd), float(planned_bet))
            reason_msg = f"REJECTED: insufficient_cash (Available: ${float(available_cash):.2f} / Needed: ${float(needed):.2f})"
            print(f"[REJECTED] {reason_msg}")
            log_func(
                "REJECTED",
                asset_type,
                token_id,
                {
                    "market_name": question,
                    "reason": "insufficient_cash",
                    "message": reason_msg,
                    "target_bet": round(float(target_bet), 4),
                    "target_bet_unclamped": round(float(target_bet_unclamped), 4),
                    "max_bet_size_usd": round(float(max_bet_size_usd), 4),
                    "bet_amount": round(float(bet_amount), 4),
                    "needed": round(float(needed), 4),
                    "available_cash": round(float(available_cash), 4),
                    "allowed_remaining": round(float(allowed_remaining), 4),
                    "daily_cap": round(float(daily_cap), 4),
                    "freed_cash": round(float(freed_cash), 4),
                    "current_balance": round(float(bridge.current_balance), 4),
                    "spent_today": round(float(spent_today), 4),
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
            bridge.current_balance = max(0.0, float(available_cash) - float(bet_amount))
            spent_today = float(spent_today) + float(bet_amount)
            bridge.spent_today = float(spent_today)
            bridge.daily_spend = float(spent_today)

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
                "max_bet_size_usd": round(float(max_bet_size_usd), 2),
                "target_bet_unclamped": round(float(target_bet_unclamped), 2),
                "target_bet_usd": round(float(target_bet), 2),
                "available_cash": round(float(available_cash), 2),
                "freed_cash": round(float(freed_cash), 2),
                "allowed_remaining": round(float(allowed_remaining), 2),
                "spent_today": round(float(spent_today), 2),
                "daily_cap": round(float(daily_cap), 2),
                "start_of_day_equity": round(float(start_of_day_equity), 2),
            },
        )
