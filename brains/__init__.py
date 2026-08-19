"""Domain-specific fair value brains: Crypto (Black-Scholes/Heston), Weather (normal), Economy (normal)."""

from .base import BaseBrain
from .crypto import HybridCryptoBrain
from .weather import WeatherBrain
from .economy import EconomyBrain

__all__ = [
    "BaseBrain",
    "HybridCryptoBrain",
    "WeatherBrain",
    "EconomyBrain",
]


def get_brain_for_asset_type(asset_type: str) -> BaseBrain:
    """Factory function to get the appropriate brain for an asset type.

    Args:
        asset_type: String like "Crypto::BTCUSDT", "Weather::Miami", "Economy::FedRate"

    Returns:
        Instantiated brain object for this asset type.

    Raises:
        ValueError: If asset_type is not recognized.
    """
    asset_type = asset_type.split("::")[0].strip().lower()

    if asset_type == "crypto":
        return HybridCryptoBrain()
    elif asset_type == "weather":
        return WeatherBrain()
    elif asset_type == "economy":
        return EconomyBrain()
    else:
        raise ValueError(f"Unknown asset type: {asset_type}")
