"""Base Benchmark Class."""

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


class BaseEvaluator(ABC):
    """Astract Class."""

    def __init__(self, logger: BaseLogger):
        self.log = logger

    @abstractmethod
    def evaluate(self, *args: Any, **kwargs: Any):
        """Abstract method for evaluation workflow of any task."""


class ForecastingEvaluator(BaseEvaluator):
    """Responsible for evaluation workflow.

    Manage prepare_metrics, prepare_estimator,
    prepare_data, validation_step, evaluate.


    strategy argument: data ingestion strategy (update, refit, no_update)
    in fitting cv

    save_checkpoints argument: save training data fitted estimator,
    each cv estimator, disable saving estimaor,
    """

    def __init__(self, logger):
        super().__init__(logger)

    def evaluate(self, *args: Any, **kwargs: Any):
        """Forecasting evaluation workflow.

        All the prepring is done in here.
        """
