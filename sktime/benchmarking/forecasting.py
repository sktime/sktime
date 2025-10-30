"""Benchmarking for forecasting estimators."""

from collections.abc import Callable

from sktime.benchmarking._benchmarking_dataclasses import (
    FoldResults,
    TaskObject,
)
from sktime.benchmarking.benchmarks import (
    BaseBenchmark,
)
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.model_evaluation import evaluate
from sktime.performance_metrics.base import BaseMetric
from sktime.split.base import BaseSplitter
from sktime.split.singlewindow import SingleWindowSplitter


class ForecastingBenchmark(BaseBenchmark):
    """Forecasting benchmark.

    Run a series of forecasters against a series of tasks defined via dataset loaders,
    cross validation splitting strategies and performance metrics, and return results as
    a df (as well as saving to file).

    Parameters
    ----------
    id_format: str, optional (default=None)
        A regex used to enforce task/estimator ID to match a certain format

    backend : string, by default "None".
        Parallelization backend to use for runs.

        - "None": executes loop sequentially, simple list comprehension
        - "loky", "multiprocessing" and "threading": uses ``joblib.Parallel`` loops
        - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``
        - "dask": uses ``dask``, requires ``dask`` package in environment
        - "dask_lazy": same as "dask", but changes the return to (lazy)
            ``dask.dataframe.DataFrame``.
        - "ray": uses ``ray``, requires ``ray`` package in environment

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

        - "ray": The following keys can be passed:

            - "ray_remote_args": dictionary of valid keys for ``ray.init``
            - "shutdown_ray": bool, default=True; False prevents ``ray`` from shutting
                down after parallelization.
            - "logger_name": str, default="ray"; name of the logger to use.
            - "mute_warnings": bool, default=False; if True, suppresses warnings

    return_data : bool, optional (default=False)
        Whether to return the prediction and the ground truth data in the results.
    """

    def add_task(
        self,
        dataset_loader: Callable | tuple,
        cv_splitter: BaseSplitter,
        scorers: list[BaseMetric],
        task_id: str | None = None,
        cv_global: BaseSplitter | None = None,
        error_score: str = "raise",
        strategy: str = "refit",
        cv_global_temporal: SingleWindowSplitter | None = None,
    ):
        """Register a forecasting task to the benchmark.

        Parameters
        ----------
        data : Union[Callable, tuple]
            Can be

            - a function which returns a dataset, like from `sktime.datasets`.
            - a tuple containing two data container that are sktime comptaible.
            - single data container that is sktime compatible (only endogenous data).

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

                forecaster.fit(y=y_train, fh=cv.fh)
                y_pred = forecaster.predict(y=y_past)
                metric(y_true, y_pred)

        error_score : "raise" or numeric, default=np.nan
            Value to assign to the score if an exception occurs in estimator fitting.
            If set to "raise", the exception is raised. If a numeric value is given,
            FitFailedWarning is raised.

        strategy : {"refit", "update", "no-update_params"}, optional, default="refit"
            defines the ingestion mode when the forecaster is updated with new data

            * "refit" = forecaster is refitted to each training window
            * "update" = forecaster is updated with training window data,
              in sequence provided
            * "no-update_params" = fit to first training window,
              re-used without fit or update

        cv_global_temporal:  SingleWindowSplitter, default=None
            ignored if cv_global is None. If passed, it splits the Panel temporally
            before the instance split from cv_global is applied. This avoids
            temporal leakage in the global evaluation across time series.
            Has to be a SingleWindowSplitter.
            cv is applied on the test set of the combined application of
            cv_global and cv_global_temporal.

        Returns
        -------
        A dictionary of benchmark results for that forecaster
        """
        if task_id is None:
            if callable(dataset_loader) and hasattr(dataset_loader, "__name__"):
                # case 1: function
                dataset_name = dataset_loader.__name__
            elif isinstance(dataset_loader, type):
                # case 2: class
                dataset_name = dataset_loader().get_tags().get("name")
            elif hasattr(dataset_loader, "get_tags"):
                # case 3: instance
                dataset_name = dataset_loader.get_tags().get("name")
            else:
                dataset_name = "_"

            task_id = (
                f"[dataset={dataset_name}]"
                + f"_[cv_splitter={cv_splitter.__class__.__name__}]"
                + (
                    f"_[cv_global={cv_global.__class__.__name__}]"
                    if cv_global is not None
                    else ""
                )
            )
        task_kwargs = {
            "data": dataset_loader,
            "cv_splitter": cv_splitter,
            "scorers": scorers,
            "cv_global": cv_global,
            "error_score": error_score,
            "cv_global_temporal": cv_global_temporal,
            "strategy": strategy,
        }
        self._add_task(
            task_id,
            TaskObject(**task_kwargs),
        )

    def _run_validation(self, task: TaskObject, estimator: BaseForecaster):
        cv_splitter = task.cv_splitter
        scorers = task.scorers
        xy_dict = task.get_y_X("forecasting")
        scores_df = evaluate(
            forecaster=estimator,
            cv=cv_splitter,
            scoring=scorers,
            backend=self.backend,
            backend_params=self.backend_params,
            error_score=task.error_score,
            return_data=self.return_data,
            cv_X=task.cv_X,
            cv_global=task.cv_global,
            strategy=task.strategy,
            return_model=False,
            cv_global_temporal=task.cv_global_temporal,
            **xy_dict,
        )

        folds = {}
        for ix, row in scores_df.iterrows():
            scores = {}
            for scorer in scorers:
                scores[scorer.name] = row["test_" + scorer.name]
            scores["fit_time"] = row["fit_time"]
            scores["pred_time"] = row["pred_time"]
            if self.return_data:
                folds[ix] = FoldResults(
                    scores, row["y_test"], row["y_pred"], row["y_train"]
                )
            else:
                folds[ix] = FoldResults(scores)
        return folds
