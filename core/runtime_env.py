"""Runtime environment / config resolution shared by every process that runs
a wallet: ui/dashboard.py (reader) and run_engine.py (the standalone trading
loop — see core/wallet_manager.py's build_wallet_runtime()).

Previously this lived only inside ui/dashboard.py (as _get_env/
_validate_runtime_env), which was fine while the engine was constructed
inline in the dashboard's page script. Now that the engine runs as its own
process, both entrypoints need to resolve config identically — this module
is the one place that happens, so there's no risk of the two processes
silently disagreeing about which wallet/config they're driving.
"""

import os

from dotenv import load_dotenv

from core.bridge import DataBridge
from core.trading_config import TradingConfig
from core.wallet_context import WalletContext
from ui import data_manager


def get_env(*names: str) -> str:
    for name in names:
        value = str(os.getenv(name, "")).strip()
        if value:
            return value
    return ""


def validate_runtime_env(bridge: DataBridge) -> WalletContext:
    """Build the single WalletContext this process drives.

    Prefers a wallet config file (WALLET_CONFIG_PATH) if one is set; falls
    back to TradingConfig.from_env() for backward compatibility with
    deployments that only set process env vars and have no wallet config
    file yet.
    """
    # Load config/.env into the process environment before checking for
    # required vars below — otherwise a fresh process (with nothing already
    # exported into the shell) always fails this check even when
    # config/.env has real values, since TradingConfig.from_env() (which
    # also calls load_dotenv) only runs after this validation passes.
    load_dotenv("config/.env")

    wallet_config_path = get_env("WALLET_CONFIG_PATH")

    if wallet_config_path:
        config = TradingConfig.from_file(wallet_config_path)
    else:
        private_key = get_env("POLYMARKET_PRIVATE_KEY", "POLYGON_PRIVATE_KEY")
        proxy_address = get_env("POLYMARKET_PROXY_ADDRESS", "POLY_ADDRESS")
        if not private_key or not proxy_address:
            raise ValueError(
                "Missing required environment variables: "
                "POLYMARKET_PRIVATE_KEY/POLYGON_PRIVATE_KEY and "
                "POLYMARKET_PROXY_ADDRESS/POLY_ADDRESS. "
                "Pass them at runtime with --env-file, or set WALLET_CONFIG_PATH "
                "to a wallet config.json."
            )
        config = TradingConfig.from_env()

    db_path = os.getenv("TRADES_DB_PATH", "/app/trades.db")
    return WalletContext(wallet_id="default", config=config, bridge=bridge, db_path=db_path)


def restore_wallet_state(ctx: WalletContext) -> dict:
    """Restore starting_balance/current_balance/start_of_day_equity/
    spent_today onto ctx.bridge from this wallet's trade history, and seed
    the live_trading DB control's default from config if no operator
    preference is on record yet.

    Called once at process startup by both ui/dashboard.py and
    run_engine.py, so a paper/dry-run restart resumes from its last known
    state instead of resetting to zero, and a fresh deploy's default trading
    mode matches its config rather than always defaulting to dry-run.

    Returns the raw restore_runtime_state() dict for callers that want the
    "source" diagnostic field too.
    """
    data_manager.init_db(ctx.db_path)
    restored = data_manager.restore_runtime_state(db_path=ctx.db_path, fallback_starting_balance=0.0)

    ctx.bridge.starting_balance = float(restored["starting_balance"])
    ctx.bridge.current_balance = float(restored["current_balance"])
    ctx.bridge.start_of_day_equity = float(restored["start_of_day_equity"])
    ctx.bridge.spent_today = float(restored["spent_today"])
    ctx.bridge.daily_spend = float(restored["spent_today"])
    ctx.bridge.state_bootstrap_source = str(restored["source"])
    if ctx.bridge.starting_balance <= 0.0:
        ctx.bridge.starting_balance = float(restored["current_balance"])

    # Config-derived default trading mode — seeds engine_control (read by
    # the engine process) only if no row exists yet (never overwrites an
    # operator's own choice on restart), and seeds this process's own
    # bridge.live_trading the same way the dashboard's toggle widget did
    # before the split (its `value=` argument only matters on first render).
    initial_live = ctx.config.is_live_run
    data_manager.seed_live_trading_requested_if_absent(ctx.db_path, initial_live)
    ctx.bridge.live_trading = initial_live

    return restored
