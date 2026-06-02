"""API clients for external data sources used by hunters."""

from abc import ABC, abstractmethod


class BaseApiClient(ABC):
    """Abstract base for all lightweight external API clients."""

    @abstractmethod
    def get_latest_value(self, *args, **kwargs):
        """Return the most recent value from the external service."""
