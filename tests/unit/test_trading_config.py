"""Tests for entry_k validation and the deprecated WANG_LAMBDA env-var guard
(see core/trading_config.py, added with the entry-side Wang redesign)."""
import logging

import pytest

from core.trading_config import TradingConfig, DEFAULT_ENTRY_K


# ── entry_k range validation (__post_init__) ────────────────────────────────

def test_entry_k_in_range_is_preserved():
    assert TradingConfig(trading_mode="dry_run", entry_k=0.3).entry_k == 0.3
    assert TradingConfig(trading_mode="dry_run", entry_k=0.0).entry_k == 0.0
    assert TradingConfig(trading_mode="dry_run", entry_k=1.0).entry_k == 1.0


def test_entry_k_above_one_is_clamped(caplog):
    with caplog.at_level(logging.WARNING):
        cfg = TradingConfig(trading_mode="dry_run", entry_k=1.5)
    assert cfg.entry_k == 1.0
    assert "entry_k" in caplog.text.lower()


def test_entry_k_below_zero_is_clamped(caplog):
    # The dangerous case: a leftover WANG_LAMBDA=-0.75 mis-mapped onto
    # entry_k. logit_shrink would treat k<=0 as total distrust (everything
    # -> 0.5); clamping to 0.0 makes that explicit and logged rather than
    # silent.
    with caplog.at_level(logging.WARNING):
        cfg = TradingConfig(trading_mode="dry_run", entry_k=-0.75)
    assert cfg.entry_k == 0.0
    assert "entry_k" in caplog.text.lower()


# ── deprecated WANG_LAMBDA env-var guard (from_env) ─────────────────────────

def test_from_env_warns_on_leftover_wang_lambda(monkeypatch, caplog):
    monkeypatch.setenv("WANG_LAMBDA", "-0.75")
    monkeypatch.delenv("ENTRY_K", raising=False)
    with caplog.at_level(logging.WARNING):
        cfg = TradingConfig.from_env()
    # Falls back to the ENTRY_K default, does NOT translate the old value.
    assert cfg.entry_k == DEFAULT_ENTRY_K
    assert "WANG_LAMBDA" in caplog.text


def test_from_env_prefers_entry_k_and_no_warning(monkeypatch, caplog):
    monkeypatch.setenv("WANG_LAMBDA", "-0.75")
    monkeypatch.setenv("ENTRY_K", "0.4")
    with caplog.at_level(logging.WARNING):
        cfg = TradingConfig.from_env()
    assert cfg.entry_k == 0.4
    # ENTRY_K is set, so the deprecation warning must NOT fire.
    assert "no longer used" not in caplog.text


def test_from_env_no_wang_lambda_no_warning(monkeypatch, caplog):
    monkeypatch.delenv("WANG_LAMBDA", raising=False)
    monkeypatch.delenv("ENTRY_K", raising=False)
    with caplog.at_level(logging.WARNING):
        cfg = TradingConfig.from_env()
    assert cfg.entry_k == DEFAULT_ENTRY_K
    assert "WANG_LAMBDA" not in caplog.text


# ── deprecated MODEL_WEIGHT env-var guard (from_env) ────────────────────────

def test_from_env_warns_on_leftover_model_weight(monkeypatch, caplog):
    # model_weight is subsumed by entry_k under the market-anchored design;
    # a leftover MODEL_WEIGHT in the .env now has no effect and should warn.
    monkeypatch.setenv("MODEL_WEIGHT", "0.40")
    with caplog.at_level(logging.WARNING):
        TradingConfig.from_env()
    assert "MODEL_WEIGHT" in caplog.text


def test_from_env_no_model_weight_no_warning(monkeypatch, caplog):
    monkeypatch.delenv("MODEL_WEIGHT", raising=False)
    with caplog.at_level(logging.WARNING):
        TradingConfig.from_env()
    assert "MODEL_WEIGHT" not in caplog.text
