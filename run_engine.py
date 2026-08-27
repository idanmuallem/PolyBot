"""Headless entrypoint for the trading engine — runs the trading loop as a
standalone OS process, independent of Streamlit and the dashboard's page
lifecycle.

Before this existed, the engine only started when a browser loaded
ui/dashboard.py (see its now-removed _ensure_engine_started_once()) —
Docker starting the container just started the Streamlit web server, which
never executes the page script until a browser connects. If no browser ever
opened the dashboard, the bot did nothing: no scanning, no trading, no
error, silently. This script removes that dependency entirely: it builds
the exact same WalletContext + runtime components the dashboard used to
build inline, via the same shared helpers ui/dashboard.py now uses too
(core.runtime_env for config resolution/state restoration,
core.wallet_manager.build_wallet_runtime for component wiring), and runs
the trading loop directly.

State is shared with the dashboard process entirely through the existing
SQLite database (trades.db, WAL mode) — see ui/data_manager.py's
engine_status/engine_control/open_positions tables. No networking, no IPC.

Per design: if this process crashes, it stays dead until the container is
restarted — no retry/supervisor logic here by design. See
config/Docker/entrypoint.sh for how this runs alongside the dashboard
process and how shutdown is propagated to both.
"""

import asyncio

from core.bridge import DataBridge
from core.runtime_env import restore_wallet_state, validate_runtime_env
from core.wallet_manager import build_wallet_runtime
from trading.decision_pipeline import run_market_monitor


def main() -> None:
    bridge = DataBridge()
    ctx = validate_runtime_env(bridge)
    restore_wallet_state(ctx)
    build_wallet_runtime(ctx)

    print(f"[ENGINE] Starting standalone trading loop (wallet_id={ctx.wallet_id!r}, db_path={ctx.db_path!r})")
    asyncio.run(run_market_monitor(ctx))


if __name__ == "__main__":
    main()
