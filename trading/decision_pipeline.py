from dataclasses import dataclass
from typing import Optional


@dataclass
class DecisionContext:
    market: object
    asset_type: str
    token_id: str
    question: str
    signal: object
    model_used: str
    poly_price: float
    actual_bet_usd: float = 0.0
    executed: bool = False
    status: str = "pending"


class DecisionHandler:
    def __init__(self):
        self._next = None

    def set_next(self, handler: "DecisionHandler") -> "DecisionHandler":
        self._next = handler
        return handler

    def handle(self, context: DecisionContext, log_func):
        result = self._process(context, log_func)
        if self._next and result is not None:
            return self._next.handle(result, log_func)
        return result

    def _process(self, context: DecisionContext, log_func):
        raise NotImplementedError


class TradabilityHandler(DecisionHandler):
    def _process(self, context: DecisionContext, log_func):
        if not context.signal.is_tradable:
            context.status = "not_tradable"
            return context
        return context


class BudgetHandler(DecisionHandler):
    def __init__(self, budget_manager):
        super().__init__()
        self.budget_manager = budget_manager

    def _process(self, context: DecisionContext, log_func):
        actual_bet_usd, should_execute = self.budget_manager.check_and_cap_bet(context.signal.kelly_size)
        if not should_execute:
            context.status = "daily_limit"
            return context
        context.actual_bet_usd = actual_bet_usd
        return context


class WatchOnlyHandler(DecisionHandler):
    def __init__(self, budget_manager):
        super().__init__()
        self.budget_manager = budget_manager

    def _process(self, context: DecisionContext, log_func):
        if self.budget_manager.watch_only:
            context.status = "watch_only"
            return context
        return context


class ExecuteHandler(DecisionHandler):
    def __init__(self, executor, budget_manager):
        super().__init__()
        self.executor = executor
        self.budget_manager = budget_manager

    def _process(self, context: DecisionContext, log_func):
        executed = self.executor.evaluate_and_execute(
            market=context.market,
            fair_value=context.signal.fair_value,
            ev=context.signal.expected_value,
            current_poly_price=context.poly_price,
            bet_amount_usd=context.actual_bet_usd,
            log_func=log_func,
        )
        context.executed = executed
        context.status = "executed" if executed else "execution_failed"
        if executed:
            self.budget_manager.record_trade(context.actual_bet_usd)
        return context


class GuardByStatusHandler(DecisionHandler):
    def __init__(self, terminal_statuses):
        super().__init__()
        self.terminal_statuses = set(terminal_statuses)

    def _process(self, context: DecisionContext, log_func):
        if context.status in self.terminal_statuses:
            return None
        return context


def build_entry_pipeline(executor, budget_manager):
    tradability = TradabilityHandler()
    stop_after_not_tradable = GuardByStatusHandler({"not_tradable"})
    budget = BudgetHandler(budget_manager)
    stop_after_budget = GuardByStatusHandler({"daily_limit"})
    watch_only = WatchOnlyHandler(budget_manager)
    stop_after_watch = GuardByStatusHandler({"watch_only"})
    execute = ExecuteHandler(executor, budget_manager)

    tradability.set_next(stop_after_not_tradable).set_next(budget).set_next(stop_after_budget).set_next(watch_only).set_next(stop_after_watch).set_next(execute)
    return tradability
