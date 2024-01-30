"""Benchmarking for forecasting estimators."""
import functools
from typing import Callable, Dict, List, Optional, Union

from sktime.benchmarking.benchmarks import BaseBenchmark
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.model_evaluation import evaluate
from sktime.performance_metrics.base import BaseMetric
from sktime.split.base import BaseSplitter


def forecasting_validation(
    dataset_loader: Callable,
    cv_splitter: BaseSplitter,
    scorers: List[BaseMetric],
    estimator: BaseForecaster,
    **kwargs,
) -> Dict[str, Union[float, str]]:
    """Run validation for a forecasting estimator.

    Parameters
    ----------
    dataset_loader : Callable
        A function which returns a dataset, like from `sktime.datasets`.
    cv_splitter : BaseSplitter object
        Splitter used for generating validation folds.
    scorers : a list of BaseMetric objects
        Each BaseMetric output will be included in the results.
    estimator : BaseForecaster object
        Estimator to benchmark.

    Returns
    -------
    Dictionary of benchmark results for that forecaster
    """
    y = dataset_loader()
    results = {}
    scores_df = evaluate(forecaster=estimator, y=y, cv=cv_splitter, scoring=scorers)
    for scorer in scorers:
        scorer_name = scorer.name
        for ix, row in scores_df.iterrows():
            results[f"{scorer_name}_fold_{ix}_test"] = row[f"test_{scorer_name}"]
        results[f"{scorer_name}_mean"] = scores_df[f"test_{scorer_name}"].mean()
        results[f"{scorer_name}_std"] = scores_df[f"test_{scorer_name}"].std()
    return results


def _factory_forecasting_validation(
    dataset_loader: Callable,
    cv_splitter: BaseSplitter,
    scorers: List[BaseMetric],
) -> Callable:
    """Build validation func which just takes a forecasting estimator."""
    return functools.partial(
        forecasting_validation,
        dataset_loader,
        cv_splitter,
        scorers,
    )


class ForecastingBenchmark(BaseBenchmark):
    """Forecasting benchmark.

    Run a series of forecasters against a series of tasks defined via dataset loaders,
    cross validation splitting strategies and performance metrics, and return results as
    a df (as well as saving to file).

    Parameters
    ----------
    id_format: str, optional (default=None)
        A regex used to enforce task/estimator ID to match a certain format

    """

    def __init__(self, id_format: Optional[str] = None):
        super().__init__(id_format)

    def add_task(
        self,
        dataset_loader: Callable,
        cv_splitter: BaseSplitter,
        scorers: List[BaseMetric],
        task_id: Optional[str] = None,
    ):
        """Register a forecasting task to the benchmark.

        Parameters
        ----------
        dataset_loader : Callable
            A function which returns a dataset, like from `sktime.datasets`.
        cv_splitter : BaseSplitter object
            Splitter used for generating validation folds.
        scorers : a list of BaseMetric objects
            Each BaseMetric output will be included in the results.
        task_id : str, optional (default=None)
            Identifier for the benchmark task. If none given then uses dataset loader
            name combined with cv_splitter class name.

        Returns
        -------
        A dictionary of benchmark results for that forecaster
        """
        task_kwargs = {
            "dataset_loader": dataset_loader,
            "cv_splitter": cv_splitter,
            "scorers": scorers,
        }
        if task_id is None:
            task_id = (
                f"[dataset={dataset_loader.__name__}]"
                f"_[cv_splitter={cv_splitter.__class__.__name__}]"
            )
        self._add_task(_factory_forecasting_validation, task_kwargs, task_id=task_id)
