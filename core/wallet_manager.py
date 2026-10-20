"""build_wallet_runtime: builds and wires one wallet's runtime components
onto an already-constructed WalletContext.

Used by run_engine.py's standalone entrypoint. WalletContext stays a plain
container — this is the one place that builds and wires what it holds.
"""

from typing import Optional, Tuple

from core.wallet_context import WalletContext
from polymarket import PolymarketScannerHunter
from trading.budget_manager import BudgetManager
from trading.clob_venue import ClobUnavailable, ClobVenue
from trading.decision_pipeline import _default_log_func, sync_live_account_state
from trading.executor import TradeExecutor
from trading.ledger import Ledger
from trading.paper_venue import PaperVenue
from trading.risk_manager import PortfolioManager
from trading.venue import Venue


def _build_venues(ctx: WalletContext) -> Tuple[Venue, Optional[Venue]]:
    """(active, standby) Venues for this wallet's configured TRADING_MODE.

    The configured mode's Venue must build: live_run without a usable CLOB
    client raises ClobUnavailable, and dry_run without a paper engine raises
    (there is no log-only mode). The other mode's Venue is built if it can
    be, so the dashboard can switch modes at runtime (the pipeline swaps the
    executor's Venue at a tick boundary); if it can't, switching to that mode
    is refused.
    """
    paper: Optional[Venue] = None
    paper_error: Optional[Exception] = None
    try:
        paper = PaperVenue.for_wallet(ctx)
        print("[PAPER] Paper trading engine initialized")
    except Exception as exc:
        paper_error = exc

    live: Optional[Venue] = None
    live_error: Optional[Exception] = None
    try:
        live = ClobVenue.from_config(ctx.config)
    except ClobUnavailable as exc:
        live_error = exc

    if ctx.config.is_live_run:
        if live is None:
            raise ClobUnavailable(f"TRADING_MODE=live_run needs a Polymarket CLOB client: {live_error}")
        if paper is None:
            print(f"[PAPER] Paper engine unavailable ({paper_error}) -- switching to paper is disabled")
        return live, paper

    if paper is None:
        raise RuntimeError(f"TRADING_MODE=dry_run needs the paper trading engine: {paper_error}") from paper_error
    if live is None:
        print(f"[LIVE] No CLOB client ({live_error}) -- switching to live is disabled")
    return paper, live


def build_wallet_runtime(ctx: WalletContext) -> None:
    """Build this wallet's runtime components in dependency order and attach
    them to ctx: executor -> scanner -> portfolio_manager -> (live balance
    sync) -> budget_manager.
    """
    ctx.ledger = Ledger(ctx.db_path)
    venue, standby_venue = _build_venues(ctx)
    ctx.venues = {v.name: v for v in (venue, standby_venue) if v is not None}
    ctx.executor = TradeExecutor(config=ctx.config, venue=venue)
    ctx.scanner = PolymarketScannerHunter(
        bridge=ctx.bridge,
        executor=ctx.executor,
        config=ctx.config,
    )
    ctx.portfolio_manager = PortfolioManager(
        bridge=ctx.bridge,
        executor=ctx.executor,
        config=ctx.config,
        hunter=ctx.scanner,
        db_path=ctx.db_path,
        ledger=ctx.ledger,
    )

    # Sync live balance before building BudgetManager so its initial_balance
    # reflects the account's real collateral, not the DataBridge default of
    # 0.0 (or, for run_engine.py's caller, whatever restore_wallet_state()
    # already restored from trade history).
    sync_live_account_state(ctx.bridge, ctx.executor, ctx.portfolio_manager, _default_log_func(ctx))

    ctx.budget_manager = BudgetManager(
        bridge=ctx.bridge,
        config=ctx.config,
        initial_balance=float(ctx.bridge.current_balance),
    )
