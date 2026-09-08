"""Force a WAL checkpoint before this container is stopped/removed during
redeploy.

trades.db runs in WAL mode (see ui/data_manager.py's init_db()) and is
bind-mounted into the container as a single file, not a directory (see
.github/workflows/deploy.yml's `docker run -v /home/ec2-user/trades.db:
/app/trades.db`). That means the *-wal sidecar holding not-yet-
checkpointed writes lives only inside *this* container's own writable
filesystem layer — never on the host. Without an explicit checkpoint,
`docker stop && docker rm` silently destroys whatever was written since
the last checkpoint, before the new container ever starts. This is what
the meta-harness's "Trade history not wiped" check (config/ci/
count_hunt_history.py) caught on a completely ordinary deploy — a real
hunt_history row count drop, not a synthetic test.

Run via `docker exec polybot python3 config/ci/wal_checkpoint.py` from
the "Deploy to EC2 via AWS SSM" step, right before `docker stop`, while
the old container is still up.
"""
import sqlite3

conn = sqlite3.connect("/app/trades.db")
conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
conn.close()
print("WAL checkpoint complete")
