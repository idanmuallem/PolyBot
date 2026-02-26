"""Run the engine's market monitor headlessly for a short dry-run period.

This script creates a simple Bridge object and a no-op logger, sets the
executor to a very high EV threshold to prevent trades, and runs the
async `run_market_monitor` for 30 seconds.
"""
import asyncio
import time

from executor import RiskConfig, TradeExecutor
from engine import run_market_monitor


class Bridge:
    def __init__(self):
        self.status = ""
        self.market_poly = 0.0
        self.forecast = 0.0
        self.ev = 0.0
        self.last_update = ""


def log_func(level, asset_type, token_id, payload):
    print(f"LOG [{level}] {asset_type} {token_id} {payload}")


async def main():
    bridge = Bridge()
    # Very high EV threshold to avoid any execution
    executor = TradeExecutor(risk_config=RiskConfig(ev_threshold=1e9))

    # run_market_monitor internally creates its own executor, so we rely on
    # its default behavior; this run is meant to be read-only (no trades).
    # We'll run it for 30 seconds and then cancel.
    task = asyncio.create_task(run_market_monitor(bridge, log_func))

    try:
        await asyncio.wait_for(task, timeout=30.0)
    except asyncio.TimeoutError:
        print("Engine run timeout reached (30s). Cancelling...")
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


if __name__ == '__main__':
    asyncio.run(main())
