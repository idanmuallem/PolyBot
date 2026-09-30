"""Abstract base class and shared utilities for all pricing brains.

Template Method pattern: evaluate() orchestrates the pipeline while subclasses
implement _calculate_probability() with domain-specific models.
"""

from abc import ABC, abstractmethod
import re
from datetime import datetime, timezone
from brains.pricing_engine import market_anchored_shrink
from core.trading_config import (
    DEFAULT_MIN_EV,
    DEFAULT_ENTRY_K,
)
from core.models import MarketData, TradeSignal


def calculate_tte(expiry_date) -> float:
    """Return time-to-expiry in days. Defaults to 30 days on parse failure."""
    default_days = 30.0
    if expiry_date is None:
        return default_days

    now = datetime.now(timezone.utc)

    if isinstance(expiry_date, datetime):
        target = expiry_date if expiry_date.tzinfo else expiry_date.replace(tzinfo=timezone.utc)
        return max(0.0, (target - now).total_seconds() / 86400.0)

    if isinstance(expiry_date, (int, float)):
        try:
            target = datetime.fromtimestamp(float(expiry_date), tz=timezone.utc)
            return max(0.0, (target - now).total_seconds() / 86400.0)
        except Exception:
            return default_days

    text = str(expiry_date).strip()
    if not text:
        return default_days

    try:
        normalized = text.replace("Z", "+00:00")
        target = datetime.fromisoformat(normalized)
        if target.tzinfo is None:
            target = target.replace(tzinfo=timezone.utc)
        return max(0.0, (target - now).total_seconds() / 86400.0)
    except Exception:
        pass

    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y", "%b %d, %Y", "%B %d, %Y"):
        try:
            target = datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            return max(0.0, (target - now).total_seconds() / 86400.0)
        except Exception:
            continue

    match = re.search(r"(20\d{2}[-/]\d{1,2}[-/]\d{1,2})", text)
    if match:
        try:
            target = datetime.strptime(
                match.group(1).replace("/", "-"), "%Y-%m-%d"
            ).replace(tzinfo=timezone.utc)
            return max(0.0, (target - now).total_seconds() / 86400.0)
        except Exception:
            pass

    return default_days


class BaseBrain(ABC):
    """Abstract base for fair value calculation brains.

    Subclasses implement _calculate_probability(); evaluate() handles EV,
    Kelly sizing, and tradability on top of that probability.
    """

    @abstractmethod
    def _calculate_probability(self, market: MarketData, live_truth: float) -> float:
        """Return a probability in [0.0, 1.0] for *market* given *live_truth*.

        *live_truth* is usually a bare spot/anchor value, but a hunter may pass
        a richer data package (e.g. CryptoHunter's CCXTDataClient dict) — see
        the concrete brain for what it accepts.
        """

    def get_raw_probability(self, market: MarketData, live_truth: float) -> float:
        """Return this brain's raw probability estimate (p_true), clamped to [0, 1].

        This is the brain's unadjusted opinion — the input evaluate() below
        risk-adjusts (Wang Transform -> market blend) into a tradeable fair
        value.
        """
        return max(0.0, min(1.0, self._calculate_probability(market, live_truth)))

    def evaluate(
        self,
        market: MarketData,
        live_truth: float,
        min_ev: float = DEFAULT_MIN_EV,
        entry_k: float = DEFAULT_ENTRY_K,
    ) -> TradeSignal:
        """Compute fair value, EV, Kelly size, and tradability for *market*.

        Single source of truth for pricing (see trading/decision_pipeline.py,
        which calls this directly rather than re-deriving fair value itself).

        Pricing is a single step: market_anchored_shrink interpolates between
        the market price and the brain's raw probability in log-odds space,
        with entry_k in [0, 1] as the trust dial (0 = defer fully to market,
        1 = trust the brain fully). This replaced a two-step design (shrink
        toward 0.5, then a separate model_weight market blend) that
        manufactured a spurious YES edge on cheap markets even when the brain
        agreed with the market - see A1_TRADE_RATE_FINDINGS.md. With the
        market-anchored form, brain == market yields post == market (zero
        edge), so the model only moves price when it genuinely diverges.
        model_weight is subsumed by entry_k and is no longer a parameter.
        """
        pre_prob = self.get_raw_probability(market, live_truth)
        market_price = float(market.initial_price) if market.initial_price > 0 else 0.5

        # Single step: trust-weighted interpolation between market and brain,
        # in logit space. entry_k is validated to [0, 1] by TradingConfig, and
        # market_anchored_shrink clamps defensively regardless.
        post_prob = market_anchored_shrink(pre_prob, market_price, entry_k)

        price_yes = market_price
        price_no = max(1e-9, 1.0 - price_yes)
        fair_no = 1.0 - post_prob
        ev_yes = (post_prob / price_yes - 1.0) if price_yes > 0 else -1.0
        ev_no = (fair_no / price_no - 1.0) if price_no > 0 else -1.0

        side = "YES" if ev_yes >= ev_no else "NO"
        expected_value = ev_yes if side == "YES" else ev_no
        entry_price = price_yes if side == "YES" else price_no

        kelly_size = max(
            0.0,
            self._calculate_kelly(post_prob, price_yes) if side == "YES"
            else self._calculate_kelly(fair_no, price_no),
        )

        is_tradable = expected_value >= min_ev and kelly_size > 0.0

        return TradeSignal(
            post_prob=post_prob,
            expected_value=expected_value,
            kelly_size=kelly_size,
            is_tradable=is_tradable,
            pre_prob=pre_prob,
            wang_fair_value=post_prob,  # no separate pre-blend value now: the
            # market-anchored interpolation IS the fair value. Field name kept
            # for storage/dashboard continuity; equals post_prob.
            wang_lambda=entry_k,  # field name kept for storage/dashboard continuity;
            # holds entry_k (trust dial, [0,1]) as of the market-anchored
            # redesign, not a probit Wang lambda. See brains/pricing_engine.py.
            wang_edge=post_prob - market_price,
            confidence=1.0,
            side=side,
            ev_yes=ev_yes,
            ev_no=ev_no,
            entry_price=entry_price,
        )

    @staticmethod
    def _calculate_kelly(fair_value: float, market_price: float) -> float:
        """Kelly criterion: (fair * (b+1) - 1) / b, where b = 1/price - 1."""
        if market_price <= 0 or market_price >= 1:
            return 0.0
        try:
            b = (1.0 / market_price) - 1.0
            if b <= 0:
                return 0.0
            return (fair_value * (b + 1.0) - 1.0) / b
        except (ValueError, ZeroDivisionError):
            return 0.0
