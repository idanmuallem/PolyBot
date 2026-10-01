import json
import logging
import os
from dataclasses import dataclass, fields

from dotenv import load_dotenv


DEFAULT_MIN_EV = 0.20
DEFAULT_MAX_BET_SIZE_USD = 3.0
DEFAULT_DAILY_LIMIT_USD = 15.0

# Entry-side pricing (see BaseBrain.evaluate() in brains/base.py). Distinct
# from wang_base_lambda below, which is the oracle3-calibrated hierarchical
# prior used only by the exit-side Wang-edge-decay check in
# trading/risk_manager.py - changing that one would silently alter open-
# position exit behavior, so entry calibration gets its own constant.
#
# entry_k is the entry-side TRUST DIAL for brains/pricing_engine.
# market_anchored_shrink(): post = sigmoid(logit(market) + k*(logit(brain) -
# logit(market))), k in [0, 1]. k=0 defers fully to the market (brain
# ignored); k=1 trusts the brain fully; in between trusts that fraction of
# the brain's divergence from the market, in logit space. When brain ==
# market, post == market for every k (no manufactured edge).
#
# This is the third entry-side design. It replaced (1) the old WANG_LAMBDA
# probit shift (crossover bug) and (2) a shrink-toward-0.5 step + separate
# model_weight blend, which manufactured a spurious YES edge on cheap markets
# even on brain/market agreement (see A1_TRADE_RATE_FINDINGS.md). entry_k
# now subsumes the old model_weight.
#
# Default (0.4) is a CONSERVATIVE, MARKET-LEANING starting value, not a fitted
# calibration. Two independent analyses (PHASE1_FINDINGS.md: r=-0.103,
# p=0.189; A1_TRADE_RATE_FINDINGS.md) found the brain's divergence from the
# market carries no usable signal yet in the traded range - the market is at
# least as well calibrated as the brain - so the model should only lightly
# influence price (defer mostly to market) until forward dry-run data, via
# scripts/forward_calibration.py, shows divergence predicts outcomes. Raise
# entry_k toward 1.0 only when that evidence appears.
DEFAULT_ENTRY_K = 0.4                # trust dial [0,1]; 0 = market only, 1 = brain only
# DEPRECATED: model_weight is subsumed by entry_k (the market-anchored
# interpolation is itself the market blend). Retained only so a deployed .env
# setting MODEL_WEIGHT doesn't error; it no longer affects pricing. from_env()
# warns if it is set. Remove after the next deploy renames the .env.
DEFAULT_MODEL_WEIGHT = 0.40          # deprecated, unused by the entry path


def _env_bool(name: str, default: str) -> bool:
    return str(os.getenv(name, default)).strip().lower() in ("true", "1", "t")


def _env_first(*names: str, default: str = "") -> str:
    for name in names:
        value = str(os.getenv(name, "")).strip()
        if value:
            return value
    return default


@dataclass
class TradingConfig:
    min_ev: float = DEFAULT_MIN_EV
    min_tte_minutes: int = 60
    max_tte_days: int = 180

    daily_limit_usd: float = DEFAULT_DAILY_LIMIT_USD
    max_bet_size_usd: float = DEFAULT_MAX_BET_SIZE_USD

    # Top-level kill switch for the arbitrage strategy path (EventSumStrategy
    # et al.), checked at the very top of SequentialTradingPipeline's
    # _stage_strategy_scan() before any Gamma API call or strategy.scan().
    # Deliberately NOT the same thing as arbitrage_max_daily_trades=0: that
    # would still scan/evaluate every cycle and just reject at the budget
    # gate, burning API calls and log noise for a strategy that should not
    # run at all. False means zero event-discovery calls, zero
    # STRATEGY-GROUP/STRATEGY-LEG log entries — the strategy simply never
    # runs, full stop.
    enable_arbitrage: bool = True

    # Per-strategy budget isolation (see trading/budget_manager.py): the
    # arbitrage and crypto-brain paths draw from the same wallet cash pool
    # but track spend/trade-count independently, so one can't starve the
    # other. daily_limit_usd/max_daily_trades above remain the global
    # ceiling — the sum of these should not exceed them (from_env/from_file
    # log a warning, not an error, if it does).
    arbitrage_daily_limit_usd: float = 50.0
    arbitrage_max_daily_trades: int = 50
    crypto_daily_limit_usd: float = 50.0
    crypto_max_daily_trades: int = 50

    # How long to let each arbitrage leg's limit order sit before giving up
    # (see TradeExecutor.execute_arbitrage_group). Zero fills on any leg
    # after this many seconds cancels the whole group — no arbitrage exists
    # without every leg.
    arbitrage_order_timeout_seconds: float = 60.0

    # Crypto event groups are higher-volume/more liquid than the general
    # Polymarket catalog, so EventSumStrategy scans them first and spends
    # whatever arbitrage budget remains on everything else (see
    # trading/strategies/event_sum.py). False restores the original
    # single-pass scan order, for A/B comparison during dry-run.
    arbitrage_crypto_first: bool = True

    bankroll_usd: float = 1000.0
    min_trading_balance: float = 5.0
    trading_mode: str = "dry_run"
    paper_balance_usd: float = 1000.0

    take_profit_pct: float = 0.20
    stop_loss_pct: float = -0.50
    min_hold_ev: float = -0.10

    loop_delay_seconds: float = 2.0
    max_daily_trades: int = 10
    private_key: str = ""
    proxy_address: str = ""
    signature_type: int = 2

    # Third-party data source keys — hunters receive these via constructor
    # injection (see hunters/weather.py, hunters/economy.py), never os.getenv().
    openweather_api_key: str = ""
    fred_api_key: str = ""

    # Wang Transform pricing (see brains/pricing_engine.py). "legacy" bypasses
    # the Wang adjustment entirely and uses the brain's raw EV, for comparing
    # the two during the transition.
    pricing_mode: str = "wang"  # "wang" or "legacy"
    wang_base_lambda: float = 0.183  # exit-side only - see trading/risk_manager.py
    wang_min_edge: float = 0.05  # minimum |wang_edge| (probability points) to consider a trade

    # Entry-side trust dial consumed by BaseBrain.evaluate() (see
    # DEFAULT_ENTRY_K above). model_weight is DEPRECATED (subsumed by entry_k)
    # and kept only for .env backward compatibility; it no longer affects
    # pricing.
    entry_k: float = DEFAULT_ENTRY_K
    model_weight: float = DEFAULT_MODEL_WEIGHT  # deprecated, unused

    # Risk management (see trading/budget_manager.py, trading/risk_manager.py).
    kelly_fraction: float = 0.25  # quarter-Kelly — full Kelly is optimal in expectation but has extreme variance
    max_drawdown_pct: float = 0.20  # equity drop from peak that pauses new entries (position exits still allowed)

    def __post_init__(self):
        if self.trading_mode not in ("dry_run", "live_run"):
            raise ValueError(
                f"TRADING_MODE must be 'dry_run' or 'live_run', got '{self.trading_mode}'"
            )

        # entry_k is a shrink strength that must live in [0, 1] (see
        # brains/pricing_engine.logit_shrink). A value outside that range is
        # almost always a mistake - most dangerously a leftover WANG_LAMBDA
        # value (e.g. -0.75) mis-mapped onto entry_k, which logit_shrink would
        # silently treat as k<=0, i.e. TOTAL distrust (every probability -> 0.5)
        # rather than the intended behavior. Clamp loudly instead of failing
        # closed, so a bad env var never silently neuters the model.
        if not (0.0 <= self.entry_k <= 1.0):
            clamped = max(0.0, min(1.0, self.entry_k))
            logging.warning(
                "entry_k=%.4f is outside [0.0, 1.0]; clamping to %.4f. If this "
                "came from a leftover WANG_LAMBDA value, note the semantics "
                "changed: ENTRY_K is a shrink strength in [0,1] (1.0 = no "
                "shrink), not a probit lambda. See brains/pricing_engine.py.",
                self.entry_k, clamped,
            )
            self.entry_k = clamped

    @property
    def is_dry_run(self) -> bool:
        return self.trading_mode == "dry_run"

    @property
    def is_live_run(self) -> bool:
        return self.trading_mode == "live_run"

    def _warn_if_strategy_budgets_exceed_ceiling(self) -> None:
        """Per-strategy budgets are independent allocations, not a hard sum
        constraint — but if they add up to more than the global ceiling,
        that's very likely a misconfiguration, so warn (don't crash)."""
        strategy_usd_total = self.arbitrage_daily_limit_usd + self.crypto_daily_limit_usd
        if strategy_usd_total > self.daily_limit_usd:
            logging.warning(
                "Per-strategy daily limits (arbitrage=$%.2f + crypto=$%.2f = $%.2f) exceed "
                "the global DAILY_LIMIT_USD ceiling ($%.2f)",
                self.arbitrage_daily_limit_usd, self.crypto_daily_limit_usd,
                strategy_usd_total, self.daily_limit_usd,
            )

        strategy_trades_total = self.arbitrage_max_daily_trades + self.crypto_max_daily_trades
        if strategy_trades_total > self.max_daily_trades:
            logging.warning(
                "Per-strategy max daily trades (arbitrage=%d + crypto=%d = %d) exceed the "
                "global MAX_DAILY_TRADES ceiling (%d)",
                self.arbitrage_max_daily_trades, self.crypto_max_daily_trades,
                strategy_trades_total, self.max_daily_trades,
            )

    @classmethod
    def from_env(cls) -> "TradingConfig":
        # Load local .env values when running outside Docker/AWS. This is a
        # side effect, so it only fires when a caller actually asks for the
        # process-env config — not on import of this module.
        load_dotenv("config/.env")

        # Deprecation guard: WANG_LAMBDA was the old entry-side env var. It is
        # no longer read (replaced by ENTRY_K, with different semantics - see
        # brains/pricing_engine.py). A deployment whose .env still sets it
        # would otherwise get no signal that the line is dead. Warn loudly so
        # the operator renames it; we deliberately do NOT translate the old
        # value, because a probit lambda and a [0,1] shrink strength are not
        # interchangeable.
        if os.getenv("WANG_LAMBDA") is not None and os.getenv("ENTRY_K") is None:
            logging.warning(
                "WANG_LAMBDA is set in the environment but is no longer used; "
                "it has been replaced by ENTRY_K (a trust dial in [0,1], 0 = "
                "market only, 1 = brain only, not a probit lambda). Falling "
                "back to ENTRY_K default (%.2f). Rename WANG_LAMBDA -> ENTRY_K "
                "in your .env to silence this.",
                DEFAULT_ENTRY_K,
            )

        # MODEL_WEIGHT is deprecated: entry_k subsumes it under the
        # market-anchored design (see brains/pricing_engine.py). Warn if a
        # deployment still sets it, since it now has no effect on pricing.
        if os.getenv("MODEL_WEIGHT") is not None:
            logging.warning(
                "MODEL_WEIGHT is set in the environment but is no longer used; "
                "the entry-side market blend is now subsumed by ENTRY_K (the "
                "market-anchored trust dial). This setting has no effect on "
                "pricing; remove it from your .env."
            )

        cfg = cls(
            min_ev=float(os.getenv("MIN_EV", str(DEFAULT_MIN_EV))),
            min_tte_minutes=int(os.getenv("MIN_TTE_MINUTES", "60")),
            max_tte_days=int(os.getenv("MAX_TTE_DAYS", "180")),
            daily_limit_usd=float(os.getenv("DAILY_LIMIT_USD", "15.0")),
            max_bet_size_usd=float(os.getenv("MAX_BET_SIZE_USD", "3.0")),
            bankroll_usd=float(os.getenv("BANKROLL_USD", "1000.0")),
            min_trading_balance=float(os.getenv("MIN_TRADING_BALANCE", "5.0")),
            take_profit_pct=float(os.getenv("TAKE_PROFIT_PCT", "0.20")),
            stop_loss_pct=float(os.getenv("STOP_LOSS_PCT", "-0.50")),
            min_hold_ev=float(os.getenv("MIN_HOLD_EV", "-0.10")),
            loop_delay_seconds=float(os.getenv("ENGINE_LOOP_DELAY", "2.0")),
            max_daily_trades=int(os.getenv("MAX_DAILY_TRADES", "10")),
            trading_mode=os.getenv("TRADING_MODE", "dry_run").strip().lower(),
            paper_balance_usd=float(os.getenv("PAPER_BALANCE_USD", "1000.0")),
            private_key=_env_first("POLYMARKET_PRIVATE_KEY", "POLYGON_PRIVATE_KEY"),
            proxy_address=_env_first("POLYMARKET_PROXY_ADDRESS", "POLY_ADDRESS"),
            signature_type=int(os.getenv("SIGNATURE_TYPE", "2")),
            openweather_api_key=_env_first("OPENWEATHER_API_KEY"),
            fred_api_key=_env_first("FRED_API_KEY"),
            pricing_mode=_env_first("PRICING_MODE", default="wang"),
            wang_base_lambda=float(os.getenv("WANG_BASE_LAMBDA", "0.183")),
            wang_min_edge=float(os.getenv("WANG_MIN_EDGE", "0.05")),
            entry_k=float(os.getenv("ENTRY_K", str(DEFAULT_ENTRY_K))),
            model_weight=float(os.getenv("MODEL_WEIGHT", str(DEFAULT_MODEL_WEIGHT))),
            kelly_fraction=float(os.getenv("KELLY_FRACTION", "0.25")),
            max_drawdown_pct=float(os.getenv("MAX_DRAWDOWN_PCT", "0.20")),
            enable_arbitrage=_env_bool("ENABLE_ARBITRAGE", "True"),
            arbitrage_daily_limit_usd=float(os.getenv("ARBITRAGE_DAILY_LIMIT_USD", "50.0")),
            arbitrage_max_daily_trades=int(os.getenv("ARBITRAGE_MAX_DAILY_TRADES", "50")),
            crypto_daily_limit_usd=float(os.getenv("CRYPTO_DAILY_LIMIT_USD", "50.0")),
            crypto_max_daily_trades=int(os.getenv("CRYPTO_MAX_DAILY_TRADES", "50")),
            arbitrage_order_timeout_seconds=float(os.getenv("ARBITRAGE_ORDER_TIMEOUT_SECONDS", "60")),
            arbitrage_crypto_first=_env_bool("ARBITRAGE_CRYPTO_FIRST", "True"),
        )
        cfg._warn_if_strategy_budgets_exceed_ceiling()
        return cfg

    @classmethod
    def from_file(cls, path: str) -> "TradingConfig":
        """Load a wallet's trading parameters from its config.json."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        defaults = cls()
        field_types = {f.name: f.type for f in fields(cls)}

        def _cast(name: str, raw_value):
            target_type = field_types.get(name)
            try:
                if target_type is bool:
                    return bool(raw_value)
                if target_type is int:
                    return int(raw_value)
                if target_type is float:
                    return float(raw_value)
            except (TypeError, ValueError):
                return getattr(defaults, name)
            return raw_value

        kwargs = {
            name: _cast(name, data[name])
            for name in field_types
            if name in data and data[name] is not None
        }
        cfg = cls(**kwargs)
        cfg._warn_if_strategy_budgets_exceed_ceiling()
        return cfg
