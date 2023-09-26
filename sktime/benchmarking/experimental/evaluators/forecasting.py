"""Interface for benchmarking experiments."""

from typing import Any, List, Optional, Union

from sktime.base import BaseEstimator
from sktime.benchmarking.base import BaseDataset
from sktime.benchmarking.experimental._base import BaseEvaluator, BaseLogger


class ForecastingEvaluator(BaseEvaluator):
    """Evaluation workflow one or more estimators on one or more datasets.

    Manage prepare_metrics, prepare_estimator,
    prepare_data, validation_step, evaluate.

    strategy argument: data ingestion strategy (update, refit, no_update)
    in fitting cv

    save_checkpoints argument: save training data fitted estimator,
    each cv estimator, disable saving estimaor,
    """

    def __init__(
        self,
        estimators: Union[BaseEstimator, List[BaseEstimator]],
        datasets: Union[BaseDataset, List[BaseDataset]],
        cv,
        logger: Optional[BaseLogger] = None,
    ):
        super().__init__(logger)

    def evaluate(self, *args: Any, **kwargs: Any):
        """Forecasting evaluation workflow.

        All the prepring is done in here.
        """
        pass

    def _prepare_data(self, *args: Any, **kwargs: Any):
        """Prepare data for forecasting evaluation."""
        pass

    def _prepare_estimator(self, *args: Any, **kwargs: Any):
        """Prepare estimator for forecasting evaluation."""
        pass

    def _prepare_metrics(self, *args: Any, **kwargs: Any):
        """Prepare metrics for forecasting evaluation."""
        pass
