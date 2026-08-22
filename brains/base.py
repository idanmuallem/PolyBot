"""Abstract base class and shared utilities for all pricing brains.

Template Method pattern: evaluate() orchestrates the pipeline while subclasses
implement _calculate_probability() with domain-specific models.
"""

from abc import ABC, abstractmethod
import logging
import re
from datetime import datetime, timezone
from brains.pricing_engine import wang_transform
from core.trading_config import (
    DEFAULT_MIN_EV,
    DEFAULT_WANG_LAMBDA,
    DEFAULT_MODEL_WEIGHT,
    DEFAULT_MAX_DISAGREEMENT_RATIO,
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
        risk-adjusts (Wang Transform -> market blend -> disagreement cap)
        into a tradeable fair value.
        """
        return max(0.0, min(1.0, self._calculate_probability(market, live_truth)))

    def evaluate(
        self,
        market: MarketData,
        live_truth: float,
        min_ev: float = DEFAULT_MIN_EV,
        wang_lambda: float = DEFAULT_WANG_LAMBDA,
        model_weight: float = DEFAULT_MODEL_WEIGHT,
        max_disagreement_ratio: float = DEFAULT_MAX_DISAGREEMENT_RATIO,
    ) -> TradeSignal:
        """Compute fair value, EV, Kelly size, and tradability for *market*.

        Single source of truth for pricing (see trading/decision_pipeline.py,
        which calls this directly rather than re-deriving fair value itself).
        Three layers run in order, each correcting what the last couldn't:

        1. Wang Transform — risk-adjusts the raw model probability. Shrinks
           toward a no-op as pre_prob approaches 0 or 1, so on its own it
           barely dents an already-overconfident (near-certain) model output.
        2. Market blending — pulls the Wang-adjusted value toward the
           market's own price by (1 - model_weight), which is what actually
           reins in a saturated raw probability the Wang step alone can't.
        3. Disagreement ratio cap — a safety net: if the blended post_prob
           still disagrees with the market by more than max_disagreement_ratio
           (either direction), the signal is rejected outright rather than
           traded on.
        """
        pre_prob = self.get_raw_probability(market, live_truth)
        market_price = float(market.initial_price) if market.initial_price > 0 else 0.5

        # Step 1: Wang Transform.
        wang_fair = wang_transform(pre_prob, wang_lambda)

        # Step 2: Market blending. model_weight is a caller-supplied knob
        # (ultimately from config/env - see MODEL_WEIGHT in core/trading_config.py),
        # so clamp defensively rather than let a bad value silently invert the
        # blend or amplify past [0, 1].
        if not (0.0 <= model_weight <= 1.0):
            logging.warning(
                "model_weight=%.4f out of [0.0, 1.0] range; clamping.", model_weight
            )
            model_weight = max(0.0, min(1.0, model_weight))
        blended_fair = model_weight * wang_fair + (1.0 - model_weight) * market_price
        post_prob = max(0.0, min(1.0, blended_fair))

        # Step 3: Disagreement ratio cap (direction-agnostic: catches
        # post_prob being either too far above or too far below the market).
        hi = max(post_prob, market_price)
        lo = max(1e-9, min(post_prob, market_price))
        disagreement_capped = (hi / lo) > max_disagreement_ratio

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

        is_tradable = (
            not disagreement_capped
            and expected_value >= min_ev
            and kelly_size > 0.0
        )

        return TradeSignal(
            post_prob=post_prob,
            expected_value=expected_value,
            kelly_size=kelly_size,
            is_tradable=is_tradable,
            pre_prob=pre_prob,
            wang_fair_value=wang_fair,
            wang_lambda=wang_lambda,
            wang_edge=post_prob - market_price,
            confidence=1.0,
            side=side,
            ev_yes=ev_yes,
            ev_no=ev_no,
            entry_price=entry_price,
            disagreement_capped=disagreement_capped,
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
