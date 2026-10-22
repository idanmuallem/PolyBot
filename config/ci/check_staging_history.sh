#!/bin/bash
# Production gate: Staging's trade history must not have shrunk since
# verify-staging recorded its baseline, and Staging must still be running
# the image Production is about to get. Shipped via config/ci/ssm_run.sh as
# the first step of the deploy-production job in .github/workflows/deploy.yml,
# with the inputs put in front of this body:
#   BASELINE=<hunt_history rows recorded by verify-staging> IMAGE=<registry/repo:sha>
#
# Fails closed: a missing baseline or an unreadable count blocks Production,
# since an unverified Staging history is not evidence that nothing was lost.
set -euo pipefail

: "${IMAGE:?IMAGE must be set}"
BASELINE="${BASELINE:-}"
CONTAINER=polybot-staging

if ! [[ "$BASELINE" =~ ^[0-9]+$ ]]; then
  echo "No usable Staging baseline from verify-staging ('$BASELINE'); cannot prove Staging history did not shrink" >&2
  exit 1
fi

# Read Config.Image from the plain JSON: SSM would take a `docker inspect
# --format` Go template's double braces for one of its own parameters.
INSPECT=$(docker inspect "$CONTAINER" 2>/dev/null || true)
if ! grep -qF "\"Image\": \"$IMAGE\"" <<<"$INSPECT"; then
  echo "Staging ($CONTAINER) is not running '$IMAGE'; Production deploys only the image Staging ran. Found:" >&2
  grep -F '"Image":' <<<"$INSPECT" >&2 || echo "  (no $CONTAINER container)" >&2
  exit 1
fi

CURRENT=$(docker exec "$CONTAINER" python3 config/ci/count_hunt_history.py | tr -d '[:space:]')
if ! [[ "$CURRENT" =~ ^[0-9]+$ ]]; then
  echo "Could not read Staging's hunt_history row count ('$CURRENT')" >&2
  exit 1
fi

echo "Staging hunt_history rows: baseline=$BASELINE now=$CURRENT"
if [ "$CURRENT" -lt "$BASELINE" ]; then
  echo "Staging hunt_history row count DROPPED ($BASELINE -> $CURRENT); trade history was lost on Staging" >&2
  exit 1
fi
echo "OK: Staging trade history did not shrink"
