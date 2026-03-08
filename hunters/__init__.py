"""
Hunters Package: Market discovery across multiple asset domains.

This package contains specialized hunters for different asset types:
- CryptoHunter: Finds cryptocurrency prediction markets (BTC, ETH, etc.)
- WeatherHunter: Finds weather prediction markets (Temperature, etc.)
- EconomyHunter: Finds economic indicator prediction markets (Fed Rate, CPI, etc.)
"""

from .base import BaseHunter
from .crypto import CryptoHunter
from .weather import WeatherHunter
from .economy import EconomyHunter
from .polymarket_scanner import PolymarketScannerHunter

__all__ = [
    "BaseHunter",
    "CryptoHunter",
    "WeatherHunter",
    "EconomyHunter",
    "PolymarketScannerHunter",
]


def get_default_hunters() -> list:
    """Get the default list of hunters in cascade order.

    Returns:
        List of instantiated hunter objects. Order determines fallback hierarchy.
    """
    return [
        CryptoHunter(),
    ]
