"""Base Benchmark Class."""

from abc import ABC, abstractmethod
from typing import Any

from sktime.benchmarking.experimental.logger import BaseLogger


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
