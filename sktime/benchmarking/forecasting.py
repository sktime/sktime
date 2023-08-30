"""Benchmarking for forecasting estimators."""
import functools
from typing import Callable, Dict, List, Optional, Union

from sktime.benchmarking.benchmarks import BaseBenchmark
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.model_selection._split import BaseSplitter
from sktime.performance_metrics.base import BaseMetric


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
    dataset_loader : Callable or a tuple
        If Callable. a function which returns a dataset, like from `sktime.datasets`
        If Tuple, must be in the format of (Y, X) where Y is the target variable
        and X is exogenous variabele where both must be sktime pd.DataFrame MTYPE.
        When tuple is given, task_id argument must be filled.
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
    # TODO:
    # dataset_loader accept sktime dataset object (future plan)
    if callable(dataset_loader):
        data = dataset_loader()
        if isinstance(data, tuple):
            y, X = data
        else:
            y, X = data, None
    else:
        y, X = dataset_loader

    results = {}
    scores_df = evaluate(
        forecaster=estimator,
        y=y,
        X=X,
        cv=cv_splitter,
        scoring=scorers,
        return_data=True,
    )
    for scorer in scorers:
        scorer_name = scorer.name
        for ix, row in scores_df.iterrows():
            results[f"{scorer_name}_fold_{ix}_test"] = row[f"test_{scorer_name}"]
            results[f"y_train_fold_{ix}"] = row["y_train"]
            results[f"y_test_fold_{ix}"] = row["y_test"]
            results[f"y_pred_fold_{ix}"] = row["y_pred"]
        results[f"{scorer_name}_mean"] = scores_df[f"test_{scorer_name}"].mean()
        results[f"{scorer_name}_std"] = scores_df[f"test_{scorer_name}"].std()

    # Alternative post-processing results form evaluate
    # for ix, row in scores_df.iterrows():
    #     results[f"y_train_fold_{ix}"] = row["y_train"]
    #     results[f"y_test_fold_{ix}"] = row["y_test"]
    #     results[f"y_pred_fold_{ix}"] = row["y_pred"]
    #     for scorer in scorers:
    #         scorer_name = scorer.name
    #         results[f"{scorer_name}_fold_{ix}_test"] = row[f"test_{scorer_name}"]

    # for scorer in scorers:
    #     scorer_name = scorer.name
    #     results[f"{scorer.name}_mean"] = scores_df[f"test_{scorer.name}"].mean()
    #     results[f"{scorer.name}_std"] = scores_df[f"test_{scorer.name}"].std()

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
    """

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
        dataset_loader : Callable or a tuple
            If Callable. a function which returns a dataset, like from `sktime.datasets`
            If Tuple, must be in the format of (Y, X) where Y is the target variable
            and X is exogenous variabele where both must be sktime pd.DataFrame MTYPE.
            When tuple is given, task_id argument must be filled.
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
        if not (callable(dataset_loader) or isinstance(dataset_loader, tuple)):
            raise TypeError("dataset_loader must be a callable or a tuple")
        task_kwargs = {
            "dataset_loader": dataset_loader,
            "cv_splitter": cv_splitter,
            "scorers": scorers,
        }
        if task_id is None:
            if isinstance(dataset_loader, tuple):
                raise ValueError(
                    "Unable to use default task_id naming. Please insert them manually"
                )
            task_id = (
                f"[dataset={dataset_loader.__name__}]"
                f"_[cv_splitter={cv_splitter.__class__.__name__}]-v1"
            )
        self._add_task(self._factory_forecasting_validation, task_kwargs, task_id)

    def validation_step(self, estimator, y, X, cv_splitter, scorers):
        """Workflows used to evaluate an estimator."""
        results = {}
        scores_df = evaluate(
            forecaster=estimator,
            y=y,
            X=X,
            cv=cv_splitter,
            scoring=scorers,
            return_data=True,
        )
        for scorer in scorers:
            scorer_name = scorer.name
            for ix, row in scores_df.iterrows():
                results[f"{scorer_name}_fold_{ix}_test"] = row[f"test_{scorer_name}"]
                results[f"y_train_fold_{ix}"] = row["y_train"]
                results[f"y_test_fold_{ix}"] = row["y_test"]
                results[f"y_pred_fold_{ix}"] = row["y_pred"]
            results[f"{scorer_name}_mean"] = scores_df[f"test_{scorer_name}"].mean()
            results[f"{scorer_name}_std"] = scores_df[f"test_{scorer_name}"].std()
        return results

    def _prepare_dataset(self, dataset_loader):
        """Prepare and validate datasets."""
        # TODO:
        # dataset_loader accept sktime dataset object (future plan)
        if callable(dataset_loader):
            data = dataset_loader()
            if isinstance(data, tuple):
                y, X = data
            else:
                y, X = data, None
        else:
            y, X = dataset_loader
        return y, X

    def _forecasting_validation(
        self,
        dataset_loader: Callable,
        cv_splitter: BaseSplitter,
        scorers: List[BaseMetric],
        estimator: BaseForecaster,
        **kwargs,
    ) -> Dict[str, Union[float, str]]:
        """Run validation for a forecasting estimator.

        Parameters
        ----------
        dataset_loader : Callable or a tuple
            If Callable. a function which returns a dataset, like from `sktime.datasets`
            If Tuple, must be in the format of (Y, X) where Y is the target variable
            and X is exogenous variabele where both must be sktime pd.DataFrame MTYPE.
            When tuple is given, task_id argument must be filled.
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
        y, X = self._prepare_dataset(dataset_loader)
        results = self.validation_step(estimator, y, X, cv_splitter, scorers)
        return results

    def _factory_forecasting_validation(
        self,
        dataset_loader: Callable,
        cv_splitter: BaseSplitter,
        scorers: List[BaseMetric],
    ) -> Callable:
        """Build validation func which just takes a forecasting estimator."""
        return functools.partial(
            self._forecasting_validation,
            dataset_loader,
            cv_splitter,
            scorers,
        )
