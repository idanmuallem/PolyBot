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
        z = (current_temp - strike_temp) / std_dev

        # Return CDF: P(T > strike) = P(Z > z_score)
        return float(norm.cdf(z))
