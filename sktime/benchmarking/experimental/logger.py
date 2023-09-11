"""Uniform interface of different experiment tracking packages."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


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


class MLFlowLogger(BaseLogger):
    """Manage the interface for tracking using Mlflow.

    Check dependencies can be handled here.
    """

    def __init__(
        self,
        experiment_name: str = "benchmark_logs",
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        save_dir: Optional[str] = "./benchmarkruns",
    ) -> None:
        pass


class WandBLogger(BaseLogger):
    """Manage the interface for tracking using WandB."""

    pass
