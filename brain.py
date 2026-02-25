# brain.py
import math
import numpy as np
from scipy.stats import norm


class QuantitativeBrain:
    """Encapsulates fair value calculations for different topic types.

    Usage:
        brain = QuantitativeBrain()
        p = brain.get_fair_value(topic_type, current_val, strike_val, time_left_days)
    """

    CRYPTO_VOL = {"BTC": 0.5, "ETH": 0.7, "SOL": 0.9}

    def get_fair_value(self, topic_type, current_val, strike_val, time_left_days, **kwargs):
        """Factory/router method selecting strategy by topic_type.

        topic_type can be like 'Crypto::BTCUSDT', 'Weather::Miami', 'Economy::FedRate'.
        Returns probability in [0.0, 1.0].
        """
        if topic_type.startswith("Crypto"):
            # Extract symbol code
            parts = topic_type.split("::")
            symbol = parts[1] if len(parts) > 1 else "BTCUSDT"
            if symbol.upper().startswith("BTC"):
                vol = self.CRYPTO_VOL.get("BTC", 0.5)
            elif symbol.upper().startswith("ETH"):
                vol = self.CRYPTO_VOL.get("ETH", 0.7)
            elif symbol.upper().startswith("SOL"):
                vol = self.CRYPTO_VOL.get("SOL", 0.9)
            else:
                vol = 0.6
            return self.calculate_crypto_prob(current_val, strike_val, time_left_days, volatility=vol)

        if topic_type.startswith("Weather"):
            # For weather, interpret current_val and strike_val as temperatures
            stddev = kwargs.get("forecast_std", 2.0)
            return self.calculate_weather_prob(current_val, strike_val, stddev, time_left_days)

        if topic_type.startswith("Economy"):
            # economy indicators often need empirical models; use simple historical vol model
            hist_vol = kwargs.get("hist_vol", 1.0)
            return self.calculate_economy_prob(current_val, strike_val, hist_vol, time_left_days)

        # Default fallback: use a conservative normal model
        return self.calculate_crypto_prob(current_val, strike_val, time_left_days, volatility=0.6)

    def calculate_crypto_prob(self, current_price, strike_price, time_to_expiry_days, volatility=0.5):
        """Black-style approximation using log-normal CDF.

        volatility is annualized (e.g. 0.5 = 50% annual vol)
        """
        if time_to_expiry_days <= 0:
            return 1.0 if current_price > strike_price else 0.0

        # annualized stdev scaled by sqrt(time)
        stdev = volatility * math.sqrt(max(1e-6, time_to_expiry_days / 365.0))
        if stdev <= 0:
            return 1.0 if current_price > strike_price else 0.0

        # d2 style term (log price ratio adjusted)
        try:
            d2 = (math.log(max(1e-12, current_price / strike_price)) - 0.5 * stdev * stdev) / stdev
        except Exception:
            return 0.5
        return float(norm.cdf(d2))

    def calculate_weather_prob(self, current_temp, strike_temp, stddev, time_to_event_days):
        """Assume forecast is normally distributed around current_temp with given stddev."""
        # If the event is immediate, treat current_temp as mean
        z = (current_temp - strike_temp) / max(1e-6, stddev)
        return float(norm.cdf(z))

    def calculate_economy_prob(self, current_val, strike_val, hist_vol, time_to_event_days):
        """Use a simple normal model on indicator changes.

        hist_vol: historical volatility in same units scaled to annual.
        """
        if time_to_event_days <= 0:
            return 1.0 if current_val > strike_val else 0.0

        stdev = hist_vol * math.sqrt(max(1e-6, time_to_event_days / 365.0))
        if stdev <= 0:
            return 1.0 if current_val > strike_val else 0.0

        z = (current_val - strike_val) / stdev
        return float(norm.cdf(z))

    @staticmethod
    def calculate_ev(market_price, fair_value):
        if market_price <= 0:
            return 0.0
        return (fair_value - market_price) / market_price