"""Benchmarking for detection estimators."""

__author__ = ["Nischal1425"]
__all__ = ["DetectionBenchmark"]

from collections.abc import Callable
from typing import Any

from sktime.benchmarking._benchmarking_dataclasses import (
    FoldResults,
    TaskObject,
)
from sktime.benchmarking.benchmarks import (
    BaseBenchmark,
)
from sktime.detection.base import BaseDetector
from sktime.detection.model_evaluation import evaluate


class DetectionBenchmark(BaseBenchmark):
    """Detection benchmark.

    Run a series of detectors against a series of tasks defined via dataset
    loaders, cross-validation splitting strategies and detection performance
    metrics, and return results as a DataFrame (as well as saving to file).

    Parameters
    ----------
    id_format: str, optional (default=None)
        A regex used to enforce task/estimator ID to match a certain format.

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
          any valid keys for ``joblib.Parallel`` can be passed here, e.g.,
          ``n_jobs``, with the exception of ``backend`` which is directly
          controlled by ``backend``. If ``n_jobs`` is not passed, it will default
          to ``-1``, other parameters will default to ``joblib`` defaults.
        - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``.
          any valid keys for ``joblib.Parallel`` can be passed here, e.g.,
          ``n_jobs``, ``backend`` must be passed as a key of ``backend_params``
          in this case. If ``n_jobs`` is not passed, it will default to ``-1``,
          other parameters will default to ``joblib`` defaults.
        - "dask": any valid keys for ``dask.compute`` can be passed,
          e.g., ``scheduler``

        - "ray": The following keys can be passed:

            - "ray_remote_args": dictionary of valid keys for ``ray.init``
            - "shutdown_ray": bool, default=True; False prevents ``ray`` from
                shutting down after parallelization.
            - "logger_name": str, default="ray"; name of the logger to use.
            - "mute_warnings": bool, default=False; if True, suppresses warnings

    return_data : bool, optional (default=False)
        Whether to return the prediction and the ground truth data in the
        results.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.model_selection import KFold
    >>> from sktime.benchmarking.detection import DetectionBenchmark
    >>> from sktime.detection.dummy import DummyRegularAnomalies
    >>> from sktime.performance_metrics.detection import WindowedF1Score

    >>> def make_data():
    ...     X = pd.DataFrame({"value": range(50)})
    ...     y = pd.DataFrame({"ilocs": [9, 19, 29, 39, 49]})
    ...     return X, y

    >>> benchmark = DetectionBenchmark()
    >>> benchmark.add_estimator(DummyRegularAnomalies(step_size=5))
    >>> benchmark.add_task(
    ...     dataset_loader=make_data,
    ...     cv_splitter=KFold(n_splits=3, shuffle=False),
    ...     scorers=[WindowedF1Score(margin=2)],
    ... )
    >>> results_df = benchmark.run()
    """

    def add_task(
        self,
        dataset_loader: Callable | tuple,
        cv_splitter: Any,
        scorers: list,
        task_id: str | None = None,
        error_score: str = "raise",
    ):
        """Register a detection task to the benchmark.

        Parameters
        ----------
        dataset_loader : Union[Callable, tuple]
            Can be

            - a function which returns a dataset, like from ``sktime.datasets``.
              For detection, should return a single time series ``X``
              (pd.DataFrame or pd.Series) or a tuple ``(X, y)`` where ``y``
              is the ground truth events DataFrame.
            - a tuple containing ``(X,)`` or ``(X, y)``.
            - a single data container (pd.DataFrame or pd.Series).

        cv_splitter : splitter object
            Splitter used for generating validation folds.
            Operates on the time index of ``X``. Recommended:

            - ``KFold(n_splits=k, shuffle=False)`` for k-fold CV
            - ``TimeSeriesSplit(n_splits=k)`` for temporal walk-forward CV

        scorers : a list of BaseDetectionMetric objects
            Each BaseDetectionMetric output will be included in the results.

        task_id : str, optional (default=None)
            Identifier for the benchmark task. If none given then uses
            dataset loader name combined with cv_splitter class name.

        error_score : "raise" or numeric, default="raise"
            Value to assign to the score if an exception occurs in estimator
            fitting. If set to ``"raise"``, the exception is raised. If a
            numeric value is given, ``FitFailedWarning`` is raised.

        Returns
        -------
        None
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
                f"_[cv_splitter={cv_splitter.__class__.__name__}]"
            )
        task_kwargs = {
            "data": dataset_loader,
            "cv_splitter": cv_splitter,
            "scorers": scorers,
            "error_score": error_score,
        }
        self._add_task(
            task_id,
            TaskObject(**task_kwargs),
        )

    def _run_validation(self, task: TaskObject, estimator: BaseDetector):
        """Run validation for a single task and estimator.

        Parameters
        ----------
        task : TaskObject
            The task object containing data, cv_splitter, and scorers.
        estimator : BaseDetector
            The detector to evaluate.

        Returns
        -------
        folds : dict of int -> FoldResults
            Results for each CV fold.
        """
        cv_splitter = task.cv_splitter
        scorers = task.scorers
        xy_dict = task.get_y_X("detection")

        scores_df = evaluate(
            detector=estimator,
            cv=cv_splitter,
            scoring=scorers,
            backend=self.backend,
            backend_params=self.backend_params,
            error_score=task.error_score,
            return_data=self.return_data,
            **xy_dict,
        )

        folds = {}
        for ix, row in scores_df.iterrows():
            scores = {}
            for scorer in scorers:
                metric_name = scorer.__class__.__name__
                scores[metric_name] = row["test_" + metric_name]
            scores["fit_time"] = row["fit_time"]
            scores["pred_time"] = row["pred_time"]
            if self.return_data:
                folds[ix] = FoldResults(
                    scores, row["y_test"], row["y_pred"], row["y_train"]
                )
            else:
                folds[ix] = FoldResults(scores)
        return folds
