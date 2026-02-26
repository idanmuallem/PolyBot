from abc import ABC, abstractmethod

class BaseApiClient(ABC):
    """Abstract base class for lightweight API clients used by hunters."""

    @abstractmethod
    def get_latest_value(self, *args, **kwargs):
        """Return the most recent value from the external service.

        The concrete signature may vary (symbol, indicator, etc.).
        """
        pass
