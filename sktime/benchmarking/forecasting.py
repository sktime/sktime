"""Benchmarking for forecasting estimators."""

import functools
from collections.abc import Callable
from typing import Optional, Union

import numpy as np

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
    cv_global=None,
    strategy="refit",
    error_score=np.nan,
    **kwargs,
) -> dict[str, Union[float, str]]:
    """Run validation for a forecasting estimator.

    Parameters
    ----------
    dataset_loader : Callable or a tuple of sktime data objects, or sktime data object

        * if sktime dataset object, must return a tuple of (y, X) upon calling ``load``,
        where ``y`` is the endogenous data container and
        ``X`` is the exogenous data container.
        * If Callable, must be a function which returns an endogenous
        data container ``y``, or a tuple with endogenous ``y`` and exogenous ``X``,
        like from ``sktime.datasets``
        * If Tuple, must be in the format of (y, X) where ``y`` is an endogenous
        data container and ``X`` is an exogenous data container.
        * if sktime data object, will be interpreted as endogenous ``y`` data container.

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

    cv_global:  sklearn splitter, or sktime instance splitter, optional, default=None
        If ``cv_global`` is passed, then global benchmarking is applied, as follows:

        1. the ``cv_global`` splitter is used to split data at instance level,
        into a global training set ``y_train``, and a global test set ``y_test_global``.
        2. The estimator is fitted to the global training set ``y_train``.
        3. ``cv_splitter`` then splits the global test set ``y_test_global`` temporally,
        to obtain temporal splits ``y_past``, ``y_true``.

        Overall, with ``y_train``, ``y_past``, ``y_true`` as above,
        the following evaluation will be applied:

        .. code-block:: python

            forecaster.fit(y=y_train, fh=cv_splitter.fh)
            y_pred = forecaster.predict(y=y_past)
            metric(y_true, y_pred)

    error_score : "raise" or numeric, default=np.nan
        Value to assign to the score if an exception occurs in estimator fitting. If set
        to "raise", the exception is raised. If a numeric value is given,
        FitFailedWarning is raised.

    strategy : {"refit", "update", "no-update_params"}, optional, default="refit"
        defines the ingestion mode when the forecaster sees new data when window expands
        "refit" = forecaster is refitted to each training window
        "update" = forecaster is updated with training window data, in sequence provided
        "no-update_params" = fit to first training window, re-used without fit or update

    Returns
    -------
    Dictionary of benchmark results for that forecaster
    """
    data = _coerce_data_for_evaluate(dataset_loader)

    scores_df = evaluate(
        forecaster=estimator,
        cv=cv_splitter,
        scoring=scorers,
        backend=backend,
        backend_params=backend_params,
        cv_global=cv_global,
        error_score=error_score,
        strategy=strategy,
        **data,  # y and X
    )

    # collect results by scorer
    results = {}
    for scorer in scorers:
        scorer_name = scorer.name
        for ix, row in scores_df.iterrows():
            results[f"{scorer_name}_fold_{ix}_test"] = row[f"test_{scorer_name}"]
        results[f"{scorer_name}_mean"] = scores_df[f"test_{scorer_name}"].mean()
        results[f"{scorer_name}_std"] = scores_df[f"test_{scorer_name}"].std()
    return results


def _coerce_data_for_evaluate(dataset_loader):
    """Coerce data input object to a dict to pass to forecasting evaluate."""
    if callable(dataset_loader) and not hasattr(dataset_loader, "load"):
        data = dataset_loader()
    elif callable(dataset_loader) and hasattr(dataset_loader, "load"):
        data = dataset_loader.load()
    else:
        data = dataset_loader

    if isinstance(data, tuple) and len(data) == 2:
        y, X = data
        return {"y": y, "X": X}
    elif isinstance(data, tuple) and len(data) == 1:
        return {"y": data[0]}
    else:
        return {"y": data}


def _factory_forecasting_validation(
    dataset_loader: Callable,
    cv_splitter: BaseSplitter,
    scorers: list[BaseMetric],
    backend=None,
    backend_params=None,
    cv_global=None,
    error_score=np.nan,
    strategy="refit",
) -> Callable:
    """Build validation func which just takes a forecasting estimator."""
    return functools.partial(
        forecasting_validation,
        dataset_loader,
        cv_splitter,
        scorers,
        backend=backend,
        backend_params=backend_params,
        cv_global=cv_global,
        error_score=error_score,
        strategy=strategy,
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
        cv_global=None,
        error_score=np.nan,
        strategy="refit",
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

        cv_global:  sklearn splitter, or sktime instance splitter, default=None
            If ``cv_global`` is passed, then global benchmarking is applied, as follows:

            1. the ``cv_global`` splitter is used to split data at instance level,
            into a global training set ``y_train``,
            and a global test set ``y_test_global``.
            2. The estimator is fitted to the global training set ``y_train``.
            3. ``cv_splitter`` then splits the global test set ``y_test_global``
            temporally, to obtain temporal splits ``y_past``, ``y_true``.

            Overall, with ``y_train``, ``y_past``, ``y_true`` as above,
            the following evaluation will be applied:

            .. code-block:: python

                forecaster.fit(y=y_train, fh=cv_splitter.fh)
                y_pred = forecaster.predict(y=y_past)
                metric(y_true, y_pred)

        error_score : "raise" or numeric, default=np.nan
            Value to assign to the score if an exception occurs in estimator fitting.
            If set to "raise", the exception is raised. If a numeric value is given,
            FitFailedWarning is raised.
        strategy : {"refit", "update", "no-update_params"}, optional, default="refit"
            defines the ingestion mode when the forecaster sees new data when window
            expands "refit" = forecaster is refitted to each training window
            "update" = forecaster is updated with training window data, in sequence
            provided "no-update_params" = fit to first training window, re-used
            without fit or update

        Returns
        -------
        A dictionary of benchmark results for that forecaster
        """
        task_kwargs = {
            "dataset_loader": dataset_loader,
            "cv_splitter": cv_splitter,
            "scorers": scorers,
            "cv_global": cv_global,
        }
        if task_id is None:
            task_id = (
                f"[dataset={dataset_loader.__name__}]"
                f"_[cv_splitter={cv_splitter.__class__.__name__}]"
            ) + (
                f"_[cv_global={cv_global.__class__.__name__}]"
                if cv_global is not None
                else ""
            )
        self._add_task(
            functools.partial(
                _factory_forecasting_validation,
                backend=self.backend,
                backend_params=self.backend_params,
                error_score=error_score,
                strategy=strategy,
            ),
            task_kwargs,
            task_id=task_id,
        )
