"""Fails if the trading engine hasn't written engine_status recently.

Run via `docker exec polybot python3 config/ci/engine_freshness_check.py`.
A container can show "Up" in `docker ps` while the engine loop inside it
has actually crashed or hung — that gap is exactly how a past outage went
undetected for days. engine_status.updated_at is written once per loop
tick by the engine process (see ui/data_manager.py's write_engine_status(),
called from trading/decision_pipeline.py); a stale timestamp means the
container is alive but the engine isn't.

Everything happens inside this one process so "now" and "updated_at" come
from the same clock — comparing across the SSM/container boundary would
risk a false mismatch from clock or timezone drift between the runner,
the host, and the container.
"""
import sys
from datetime import datetime

from ui.data_manager import read_engine_status

DB_PATH = "/app/trades.db"
STALE_AFTER_SECONDS = 150  # "a couple of minutes" — loop_delay_seconds defaults to 2s


def main() -> int:
    status = read_engine_status(DB_PATH)
    if not status or not status.get("updated_at"):
        print("FAIL: no engine_status row found (engine has never written status)")
        return 1

    updated_at = datetime.strptime(status["updated_at"], "%Y-%m-%d %H:%M:%S")
    age_seconds = (datetime.now() - updated_at).total_seconds()

    if age_seconds > STALE_AFTER_SECONDS:
        print(
            f"FAIL: engine_status.updated_at is {age_seconds:.0f}s old "
            f"(threshold {STALE_AFTER_SECONDS}s) — the engine process looks "
            f"stuck or dead even though the container is Up"
        )
        return 1

    print(f"OK: engine_status.updated_at is {age_seconds:.0f}s old")
    return 0


if __name__ == "__main__":
    sys.exit(main())
