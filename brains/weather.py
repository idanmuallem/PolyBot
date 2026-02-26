"""
WeatherBrain: Fair value calculation for weather prediction markets.

Uses a normal distribution model with configurable standard deviation.
"""

from scipy.stats import norm

from .base import BaseBrain


class WeatherBrain(BaseBrain):
    """Calculate fair value for weather prediction markets.

    Model: Normal distribution around forecast with configurable std dev.
    Factors: Current temperature, strike temperature, forecast uncertainty.
    """

    DEFAULT_STD_DEV = 2.0  # Celsius degrees

    def __init__(self, std_dev: float = DEFAULT_STD_DEV):
        """Initialize WeatherBrain.

        Args:
            std_dev: Standard deviation of temperature forecast (default: 2°C)
        """
        self.std_dev = std_dev

    def get_topic_type(self) -> str:
        return "Weather"

    def get_fair_value(
        self,
        live_truth: float,
        strike: float,
        days_left: float,
        **kwargs
    ) -> float:
        """Calculate fair value using normal distribution.

        For weather, we assume the current temperature is the mean estimate,
        and use the standard deviation to model uncertainty.

        Args:
            live_truth: Current temperature (°C)
            strike: Strike temperature (°C)
            days_left: Not used in this model (weather is typically short-term)
            **kwargs: Override std_dev with 'std_dev' kwarg if provided

        Returns:
            Probability (CDF value) in [0.0, 1.0]
        """
        std_dev = kwargs.get("std_dev", self.std_dev)

        # Support range-based strikes: if 'strike_low' and/or 'strike_high' provided,
        # compute interval probability. For open-ended 'above' ranges, provide
        # 'strike_low' with 'strike_high' == None.
        strike_low = kwargs.get("strike_low")
        strike_high = kwargs.get("strike_high")

        if strike_low is not None and strike_high is not None:
            return self._calculate_prob_range(live_truth, strike_low, strike_high, std_dev)
        if strike_low is not None and strike_high is None:
            return self._calculate_prob_above(live_truth, strike_low, std_dev)

        return self._calculate_prob(live_truth, strike, std_dev)

    @staticmethod
    def _calculate_prob(
        current_temp: float,
        strike_temp: float,
        std_dev: float = 2.0
    ) -> float:
        """Calculate probability using normal CDF.

        Computes P(temp > strike) assuming temp ~ N(current_temp, std_dev²)

        Args:
            current_temp: Current observed temperature
            strike_temp: Strike/threshold temperature
            std_dev: Standard deviation of the distribution

        Returns:
            Probability in [0.0, 1.0]
        """
        if std_dev <= 0:
            std_dev = 0.1  # Avoid division by zero

        # Z-score
        z = (strike_temp - current_temp) / std_dev

        # Return P(T > strike) = 1 - CDF(z)
        return float(1.0 - norm.cdf(z))

    @staticmethod
    def _calculate_prob_range(
        current_temp: float,
        strike_low: float,
        strike_high: float,
        std_dev: float = 2.0
    ) -> float:
        """Calculate probability that temperature falls within [strike_low, strike_high].

        Uses P(low <= T <= high) = CDF((high-mean)/sd) - CDF((low-mean)/sd).
        """
        if std_dev <= 0:
            std_dev = 0.1

        low_z = (strike_low - current_temp) / std_dev
        high_z = (strike_high - current_temp) / std_dev
        return float(norm.cdf(high_z) - norm.cdf(low_z))

    @staticmethod
    def _calculate_prob_above(
        current_temp: float,
        strike_low: float,
        std_dev: float = 2.0
    ) -> float:
        """Calculate probability that temperature is above strike_low.

        Uses P(T > strike_low) = 1 - CDF((strike_low - mean)/sd).
        """
        if std_dev <= 0:
            std_dev = 0.1
        z = (strike_low - current_temp) / std_dev
        return float(1.0 - norm.cdf(z))
