#!/bin/bash
# Meta-harness check: does the live container's resolved TradingConfig
# actually reflect what's in /home/ec2-user/.env (the file `docker run
# --env-file` was pointed at) right now? Catches config drift where the
# running container's env has fallen out of sync with the .env file on
# disk — e.g. an operator restarting via `docker start` instead of the
# full stop/rm/run recreation deploy.yml does, which would leave the
# container's env baked in from whenever it was first created.
#
# Runs on the EC2 host itself (shipped via config/ci/ssm_run.sh from
# .github/workflows/deploy.yml) so it can read both sides: the host's env
# file and, via `docker exec`, the container's actual resolved config.
set -euo pipefail

ENV_FILE="/home/ec2-user/.env"
FAIL=0

LIVE=$(docker exec polybot python3 config/ci/config_readback.py)

# Non-secret knobs worth checking. Deliberately a short, curated list —
# meant to catch drift at a glance, not replicate every TradingConfig
# field (most of the others rarely change and would just add noise).
check_key() {
  env_key="$1"
  cfg_key="$2"
  # `|| true` on both: under `set -o pipefail`, grep finding no match (an
  # unset key in .env, or an unexpected miss on the live side) makes the
  # whole pipeline "fail", which would abort the script right here via
  # `set -e` before the empty-expected handling below ever runs.
  expected=$(grep -E "^${env_key}=" "$ENV_FILE" 2>/dev/null | tail -n1 | cut -d= -f2- || true)
  live=$(printf '%s\n' "$LIVE" | grep -E "^${cfg_key}=" | cut -d= -f2- || true)

  if [ -z "$expected" ]; then
    echo "SKIP ${env_key}: not set in ${ENV_FILE} (code default in use)"
    return 0
  fi

  norm_expected=$(printf '%s' "$expected" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')
  norm_live=$(printf '%s' "$live" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')
  if [ "$norm_expected" = "$norm_live" ]; then
    echo "OK ${env_key}=${live}"
    return 0
  fi

  # Numeric fallback so e.g. "15" in .env vs "15.0" in the resolved config
  # (a float field) doesn't false-positive as a mismatch.
  if python3 -c "import sys; sys.exit(0 if abs(float('$expected') - float('$live')) < 1e-9 else 1)" 2>/dev/null; then
    echo "OK ${env_key}=${live} (numerically equal to .env's ${expected})"
    return 0
  fi

  echo "MISMATCH ${env_key}: .env='${expected}' but running container resolved ${cfg_key}='${live}'"
  FAIL=1
}

check_key TRADING_MODE trading_mode
check_key MIN_EV min_ev
check_key DAILY_LIMIT_USD daily_limit_usd
check_key MAX_BET_SIZE_USD max_bet_size_usd
check_key BANKROLL_USD bankroll_usd
check_key KELLY_FRACTION kelly_fraction
check_key MAX_DRAWDOWN_PCT max_drawdown_pct
check_key ENABLE_ARBITRAGE enable_arbitrage

# CI VERIFICATION (deliberate failure): forces a mismatch to prove this
# check actually fails and is reported clearly. Will revert immediately
# after confirming. See .github/workflows/deploy.yml.
echo "MISMATCH CI_VERIFICATION: forced mismatch to prove this check fails and is caught"
FAIL=1

exit "$FAIL"
