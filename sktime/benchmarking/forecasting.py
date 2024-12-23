"""Benchmarking for forecasting estimators."""

import functools
from collections.abc import Callable
from typing import Optional, Union

from sktime.benchmarking.benchmarks import BaseBenchmark
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.model_evaluation import evaluate
from sktime.performance_metrics.base import BaseMetric
from sktime.split.base import BaseSplitter


def forecasting_validation(
    dataset_loader: Callable,
    cv_splitter: BaseSplitter,
    scorers: list[BaseMetric],
    estimator: BaseForecaster,
    backend=None,
    backend_params=None,
    global_mode=False,
    cv_ht=None,
    **kwargs,
) -> dict[str, Union[float, str]]:
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
    backend : {"dask", "loky", "multiprocessing", "threading"}, by default None.
        Runs parallel evaluate for each task if specified.

        - "None": executes loop sequentally, simple list comprehension
        - "loky", "multiprocessing" and "threading": uses ``joblib.Parallel`` loops
        - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``
        - "dask": uses ``dask``, requires ``dask`` package in environment
        - "dask_lazy": same as "dask",
        but changes the return to (lazy) ``dask.dataframe.DataFrame``.

        Recommendation: Use "dask" or "loky" for parallel evaluate.
        "threading" is unlikely to see speed ups due to the GIL and the serialization
        backend (``cloudpickle``) for "dask" and "loky" is generally more robust
        than the standard ``pickle`` library used in "multiprocessing".

    backend_params : dict, optional
        additional parameters passed to the backend as config.
        Directly passed to ``utils.parallel.parallelize``.
        Valid keys depend on the value of ``backend``:

        - "None": no additional parameters, ``backend_params`` is ignored
        - "loky", "multiprocessing" and "threading": default ``joblib`` backends
        any valid keys for ``joblib.Parallel`` can be passed here, e.g., ``n_jobs``,
        with the exception of ``backend`` which is directly controlled by ``backend``.
        If ``n_jobs`` is not passed, it will default to ``-1``, other parameters
        will default to ``joblib`` defaults.
        - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``.
        any valid keys for ``joblib.Parallel`` can be passed here, e.g., ``n_jobs``,
        ``backend`` must be passed as a key of ``backend_params`` in this case.
        If ``n_jobs`` is not passed, it will default to ``-1``, other parameters
        will default to ``joblib`` defaults.
        - "dask": any valid keys for ``dask.compute`` can be passed,
        e.g., ``scheduler``
    global_mode: bool, default=False
        If `global_mode=True`, following changes are applied:
        `cv_splitter` is used to split data in instance level.
        `cv_splitter` must be an instance of `InstanceSplitter`.
        `cv_ht` is used to split `y_test` from `cv_splitter`
        to `y_hist` and `y_true` in time index level.
        `cv_ht` must be an instance of `SingleWindowSplitter`.
        With `y_train`, `y_hist`, `y_true`, `X_train`, `X_test`
        from each fold, following evaluation will be applied:
        ```py
        forecaster.fit(y=y_train, X=X_train, fh=cv_ht.fh)
        y_pred = forecaster.predict((y=y_hist, X=X_test)
        # calculate metrics with `y_true` and `y_pred`
        ```
    cv_ht : sktime InstanceSplitter descendant, optional
        In global mode, `cv_ht` is used to split `y_test` to 'y_hist and 'y_true`

    Returns
    -------
    Dictionary of benchmark results for that forecaster
    """
    data = dataset_loader()
    if isinstance(data, tuple):
        X, y = data
    else:
        X = None
        y = data

    results = {}
    if isinstance(y, tuple):
        y, X = y
        scores_df = evaluate(
            forecaster=estimator,
            y=y,
            X=X,
            cv=cv_splitter,
            scoring=scorers,
            backend=backend,
            backend_params=backend_params,
            global_mode=global_mode,
            cv_ht=cv_ht,
        )
    else:
        scores_df = evaluate(
            forecaster=estimator,
            y=y,
            cv=cv_splitter,
            scoring=scorers,
            backend=backend,
            backend_params=backend_params,
            global_mode=global_mode,
            cv_ht=cv_ht,
        )

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
    scorers: list[BaseMetric],
    backend=None,
    backend_params=None,
    global_mode=False,
    cv_ht=None,
) -> Callable:
    """Build validation func which just takes a forecasting estimator."""
    return functools.partial(
        forecasting_validation,
        dataset_loader,
        cv_splitter,
        scorers,
        backend=backend,
        backend_params=backend_params,
        global_mode=global_mode,
        cv_ht=cv_ht,
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
    backend : {"dask", "loky", "multiprocessing", "threading"}, by default None.
        Runs parallel evaluate for each task if specified.

        - "None": executes loop sequentally, simple list comprehension
        - "loky", "multiprocessing" and "threading": uses ``joblib.Parallel`` loops
        - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``
        - "dask": uses ``dask``, requires ``dask`` package in environment
        - "dask_lazy": same as "dask",
        but changes the return to (lazy) ``dask.dataframe.DataFrame``.

        Recommendation: Use "dask" or "loky" for parallel evaluate.
        "threading" is unlikely to see speed ups due to the GIL and the
        serialization backend (``cloudpickle``) for "dask" and "loky" is
        generally more robust than the standard ``pickle`` library used
        in "multiprocessing".

    backend_params : dict, optional
        additional parameters passed to the backend as config.
        Directly passed to ``utils.parallel.parallelize``.
        Valid keys depend on the value of ``backend``:

        - "None": no additional parameters, ``backend_params`` is ignored
        - "loky", "multiprocessing" and "threading": default ``joblib`` backends
        any valid keys for ``joblib.Parallel`` can be passed here, e.g., ``n_jobs``,
        with the exception of ``backend`` which is directly controlled by
        ``backend``. If ``n_jobs`` is not passed, it will default to ``-1``, other
        parameters will default to ``joblib`` defaults.
        - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``.
        any valid keys for ``joblib.Parallel`` can be passed here, e.g., ``n_jobs``,
        ``backend`` must be passed as a key of ``backend_params`` in this case.
        If ``n_jobs`` is not passed, it will default to ``-1``, other parameters
        will default to ``joblib`` defaults.
        - "dask": any valid keys for ``dask.compute`` can be passed,
        e.g., ``scheduler``
    """

    def __init__(
        self,
        id_format: Optional[str] = None,
        backend=None,
        backend_params=None,
    ):
        super().__init__(id_format)
        self.backend = backend
        self.backend_params = backend_params

    def add_task(
        self,
        dataset_loader: Callable,
        cv_splitter: BaseSplitter,
        scorers: list[BaseMetric],
        task_id: Optional[str] = None,
        global_mode=False,
        cv_ht=None,
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
        global_mode: bool, default=False
            If `global_mode=True`, following changes are applied:
            `cv_splitter` is used to split data in instance level.
            `cv_splitter` must be an instance of `InstanceSplitter`.
            `cv_ht` is used to split `y_test` from `cv_splitter`
            to `y_hist` and `y_true` in time index level.
            `cv_ht` must be an instance of `SingleWindowSplitter`.
            With `y_train`, `y_hist`, `y_true`, `X_train`, `X_test`
            from each fold, following evaluation will be applied:
            ```py
            forecaster.fit(y=y_train, X=X_train, fh=cv_ht.fh)
            y_pred = forecaster.predict((y=y_hist, X=X_test)
            # calculate metrics with `y_true` and `y_pred`
            ```
        cv_ht : sktime InstanceSplitter descendant, optional
            In global mode, `cv_ht` is used to split `y_test` to 'y_hist and 'y_true`

        Returns
        -------
        A dictionary of benchmark results for that forecaster
        """
        task_kwargs = {
            "dataset_loader": dataset_loader,
            "cv_splitter": cv_splitter,
            "scorers": scorers,
            "global_mode": global_mode,
            "cv_ht": cv_ht,
        }
        if task_id is None:
            task_id = (
                f"[dataset={dataset_loader.__name__}]"
                f"_[cv_splitter={cv_splitter.__class__.__name__}]"
            ) + (f"_[cv_ht={cv_ht.__class__.__name__}]" if global_mode else "")
        self._add_task(
            functools.partial(
                _factory_forecasting_validation,
                backend=self.backend,
                backend_params=self.backend_params,
            ),
            task_kwargs,
            task_id=task_id,
        )
