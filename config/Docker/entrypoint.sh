#!/bin/sh
# Runs both processes in one container: the trading engine (headless,
# run_engine.py) and the Streamlit dashboard, sharing state only through
# trades.db (WAL mode — see ui/data_manager.py). No supervisor daemon: this
# script is the whole of it.
#
# The engine runs in the background; Streamlit is the foreground process so
# it receives `docker stop`'s SIGTERM directly. The trap below forwards that
# signal to the backgrounded engine process too — without it, the engine
# would be orphaned (reparented, unsignaled) and only cleaned up whenever
# Docker's stop-timeout expires and force-kills the whole container. This is
# shutdown propagation only, not crash recovery: if the engine process dies
# on its own, it stays dead until the container is restarted — no retry
# logic here, by design.
set -e

python run_engine.py &
ENGINE_PID=$!

term_handler() {
  kill -TERM "$ENGINE_PID" 2>/dev/null
  kill -TERM "$STREAMLIT_PID" 2>/dev/null
}
trap term_handler TERM INT

streamlit run ui/dashboard.py --server.port=8501 --server.address=0.0.0.0 &
STREAMLIT_PID=$!
wait "$STREAMLIT_PID"
