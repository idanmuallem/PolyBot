#!/bin/bash
# Replaces the Staging container on the EC2 host with the given image.
# Shipped via config/ci/ssm_run.sh from the deploy-staging job in
# .github/workflows/deploy.yml, with the inputs put in front of this body:
#   REGISTRY=<ecr registry> IMAGE=<registry/repo:sha> AWS_REGION=<region>
#
# Staging is a second container next to Production on the same instance:
#   - container `polybot-staging`, dashboard on host port 8502
#   - its own env file (no wallet key), with DEPLOYMENT_ENV=staging forced
#     on top so the Staging lock (core/runtime_env.py) always applies and a
#     Live run TRADING_MODE stops it starting
#   - its own trade database and paper directory, wiped on every deploy so
#     each version is judged on its own run. Both are bind-mounted from the
#     host and the container restarts in place (--restart always), so they
#     survive a container restart within a run.
# Production's container, files and env file are never touched here.
set -euo pipefail

: "${REGISTRY:?REGISTRY must be set}"
: "${IMAGE:?IMAGE must be set}"
: "${AWS_REGION:?AWS_REGION must be set}"

CONTAINER=polybot-staging
ENV_FILE=/home/ec2-user/.env.staging
DATA_DIR=/home/ec2-user/staging

if [ ! -f "$ENV_FILE" ]; then
  echo "Staging env file $ENV_FILE is missing (see README, Deployment > Staging)" >&2
  exit 1
fi
# Staging holds no wallet key. Refuse to deploy rather than run with one.
if grep -qE '^[[:space:]]*(export[[:space:]]+)?(POLYMARKET_PRIVATE_KEY|POLYGON_PRIVATE_KEY)[[:space:]]*=[[:space:]]*[^[:space:]]' "$ENV_FILE"; then
  echo "Staging env file $ENV_FILE sets a wallet private key; Staging must hold none" >&2
  exit 1
fi

aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$REGISTRY"
docker pull "$IMAGE"

docker stop "$CONTAINER" || true
docker rm "$CONTAINER" || true

# Fresh trade history and paper state for this run.
rm -rf "$DATA_DIR"
mkdir -p "$DATA_DIR/paper_trading"
touch "$DATA_DIR/trades.db"
chmod 666 "$DATA_DIR/trades.db"
chmod 777 "$DATA_DIR/paper_trading"

docker run -d --name "$CONTAINER" --restart always \
  --log-opt max-size=20m --log-opt max-file=3 \
  -p 8502:8501 \
  -v "$DATA_DIR/trades.db:/app/trades.db" \
  -v "$DATA_DIR/paper_trading:/app/paper_trading" \
  --env-file "$ENV_FILE" \
  -e DEPLOYMENT_ENV=staging \
  "$IMAGE"

# Removes only images no container uses: Production's image stays.
docker image prune -a -f

echo "Staging deployed: $IMAGE"
