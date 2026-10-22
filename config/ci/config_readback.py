"""Prints the live container's resolved TradingConfig, one KEY=value per
line, with secret-bearing fields redacted.

Run via `docker exec polybot python3 config/ci/config_readback.py`. This
calls the exact same TradingConfig.from_env() the app itself builds at
startup — printing it from *inside the running container* (not from the
repo or from the .env file on the host) is what catches config drift like
`docker start` reusing a stale env instead of the current config/.env (the
"docker start didn't reload .env" class of bug). See
config/ci/check_config_drift.sh, which diffs this against the host's env
file, and .github/workflows/deploy.yml.
"""
from dataclasses import asdict

from core.trading_config import TradingConfig, deployment_environment

REDACT_FIELDS = {"private_key", "openweather_api_key", "fred_api_key"}


def main() -> None:
    cfg = asdict(TradingConfig.from_env())
    # The environment marker is a deployment setting, not a TradingConfig
    # field, but drift checks need to see which environment the container
    # believes it is (an unset DEPLOYMENT_ENV reads as production).
    cfg["deployment_env"] = deployment_environment()
    for key in sorted(cfg):
        value = "<redacted>" if key in REDACT_FIELDS else cfg[key]
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
