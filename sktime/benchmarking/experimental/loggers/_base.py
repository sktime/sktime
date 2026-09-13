"""Base class for loggers."""

from abc import ABC, abstractmethod
from typing import Any


class BaseLogger(ABC):
    """Abstact class that provides blueprint for logger classes."""

    @abstractmethod
    def log_metrics(self, *args: Any, **kwargs: Any) -> None:
        """Log scores."""
        pass

    @abstractmethod
    def log_hyperparams(self, *args: Any, **kwargs: Any) -> None:
        """Log estimator parameters."""
        pass

    @abstractmethod
    def log_estimator(self, *args: Any, **kwargs: Any) -> None:
        """Save estimator state."""
        pass

    @abstractmethod
    def log_graph(self, *args: Any, **kwargs: Any) -> None:
        """For logging visualisation."""
        pass

    @abstractmethod
    def log_cvsplit(self, *args: Any, **kwargs: Any) -> None:
        """For logging cvsplit metadata.

        Log cv split metadata such as cuttoff time, len, number of splits
        I think this is a better aproach than saving them as multiple small
        dataset chuncks. Managing datasets is delegated to dataset class so that
        we have more control?
        """
        pass

    @abstractmethod
    def start_run(self, *args: Any, **kwargs: Any) -> None:
        """Start a new active run."""
        pass

    @abstractmethod
    def end_run(self, *args: Any, **kwargs: Any) -> None:
        """End an active run."""
        pass
