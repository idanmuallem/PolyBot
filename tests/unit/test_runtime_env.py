"""Startup validation (core/runtime_env.validate_runtime_env): which
Dry run / Live run and Staging / Production combinations may start, driven
only by environment variables."""
import pytest

from core.bridge import DataBridge
from core.runtime_env import validate_runtime_env

_ENV_KEYS = (
    "WALLET_CONFIG_PATH",
    "POLYMARKET_PRIVATE_KEY",
    "POLYGON_PRIVATE_KEY",
    "POLYMARKET_PROXY_ADDRESS",
    "POLY_ADDRESS",
    "TRADING_MODE",
    "DEPLOYMENT_ENV",
)


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    # Both modules call load_dotenv("config/.env"), which would re-inject a
    # developer's local key or trading mode. Each test sets exactly the env
    # it needs.
    monkeypatch.setattr("core.runtime_env.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("core.trading_config.load_dotenv", lambda *a, **k: None)
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _set_wallet_key(monkeypatch):
    monkeypatch.setenv("POLYMARKET_PRIVATE_KEY", "0xabc")
    monkeypatch.setenv("POLYMARKET_PROXY_ADDRESS", "0xdef")


def test_dry_run_starts_without_wallet_key_or_proxy_address(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "dry_run")

    ctx = validate_runtime_env(DataBridge())

    assert ctx.config.is_dry_run
    assert ctx.config.private_key == ""


def test_live_run_without_wallet_key_refuses_to_start(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "live_run")
    monkeypatch.setenv("POLYMARKET_PROXY_ADDRESS", "0xdef")

    with pytest.raises(ValueError, match="Missing required environment variables"):
        validate_runtime_env(DataBridge())


def test_staging_refuses_to_start_in_live_run_even_with_a_wallet_key(monkeypatch):
    monkeypatch.setenv("DEPLOYMENT_ENV", "staging")
    monkeypatch.setenv("TRADING_MODE", "live_run")
    _set_wallet_key(monkeypatch)

    with pytest.raises(ValueError, match="Staging is locked to Dry run"):
        validate_runtime_env(DataBridge())


def test_keyless_staging_in_live_run_names_the_staging_lock(monkeypatch):
    monkeypatch.setenv("DEPLOYMENT_ENV", "staging")
    monkeypatch.setenv("TRADING_MODE", "live_run")

    with pytest.raises(ValueError, match="Staging is locked to Dry run"):
        validate_runtime_env(DataBridge())


def test_staging_starts_keyless_in_dry_run(monkeypatch):
    monkeypatch.setenv("DEPLOYMENT_ENV", "staging")
    monkeypatch.setenv("TRADING_MODE", "dry_run")

    ctx = validate_runtime_env(DataBridge())

    assert ctx.config.is_dry_run


@pytest.mark.parametrize("marker", [None, "production"])
def test_unset_marker_behaves_as_production_live_run_with_key_starts(monkeypatch, marker):
    if marker is not None:
        monkeypatch.setenv("DEPLOYMENT_ENV", marker)
    monkeypatch.setenv("TRADING_MODE", "live_run")
    _set_wallet_key(monkeypatch)

    ctx = validate_runtime_env(DataBridge())

    assert ctx.config.is_live_run


@pytest.mark.parametrize("marker", [None, "production"])
def test_unset_marker_behaves_as_production_live_run_without_key_refused(monkeypatch, marker):
    if marker is not None:
        monkeypatch.setenv("DEPLOYMENT_ENV", marker)
    monkeypatch.setenv("TRADING_MODE", "live_run")

    with pytest.raises(ValueError, match="Missing required environment variables"):
        validate_runtime_env(DataBridge())


def test_unknown_deployment_environment_refuses_to_start(monkeypatch):
    monkeypatch.setenv("DEPLOYMENT_ENV", "prod")
    monkeypatch.setenv("TRADING_MODE", "dry_run")

    with pytest.raises(ValueError, match="DEPLOYMENT_ENV"):
        validate_runtime_env(DataBridge())
