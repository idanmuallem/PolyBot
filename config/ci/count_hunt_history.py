"""Prints the current hunt_history row count (0 if the table doesn't exist).

Run via `docker exec polybot python3 config/ci/count_hunt_history.py` —
always from inside the running container, never against the host-side
trades.db file directly: trades.db is opened in WAL mode
(ui/data_manager.py) and bind-mounted into the container as a single file
(see .github/workflows/deploy.yml's `docker run -v`), so the *-wal sidecar
holding not-yet-checkpointed writes lives only in the container's
filesystem — reading the host copy directly can undercount.

Used by both the pre-deploy baseline snapshot and the post-deploy
meta-harness comparison, so a dropped count (instead of flat-or-growing)
between the two flags a wipe.
"""
import sqlite3

conn = sqlite3.connect("/app/trades.db")
exists = conn.execute(
    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='hunt_history'"
).fetchone()
print(conn.execute("SELECT COUNT(*) FROM hunt_history").fetchone()[0] if exists else 0)
