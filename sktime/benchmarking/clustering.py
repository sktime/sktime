"""Benchmarking for clustering estimators."""

__author__ = ["Nischal1425"]
__all__ = ["ClusteringBenchmark"]

from collections.abc import Callable
from typing import Any

from sktime.benchmarking._benchmarking_dataclasses import (
    FoldResults,
    TaskObject,
)
from sktime.benchmarking.benchmarks import (
    BaseBenchmark,
)
from sktime.clustering.base import BaseClusterer
from sktime.clustering.model_evaluation import evaluate


class ClusteringBenchmark(BaseBenchmark):
    """Clustering benchmark.

    Run a series of clusterers against a series of tasks defined via dataset loaders,
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
        cv_splitter: Any,
        scorers: list,
        task_id: str | None = None,
        error_score: str = "raise",
    ):
        """Register a clustering task to the benchmark.

        Parameters
        ----------
        data : Union[Callable, tuple]
            Can be
            - a function which returns a dataset, like from `sktime.datasets`.
            - a tuple containing two data container that are sktime compatible.
            - single data container that is sktime compatible (only panel data).
        cv_splitter : BaseSplitter object
            Splitter used for generating validation folds.
        scorers : a list of callable metric objects
            Each metric output will be included in the results.
            Supports both internal metrics (e.g., ``silhouette_score``) and
            external metrics (e.g., ``adjusted_rand_score``).
        task_id : str, optional (default=None)
            Identifier for the benchmark task. If none given then uses dataset loader
            name combined with cv_splitter class name.
        error_score : "raise" or numeric, default=np.nan
            Value to assign to the score if an exception occurs in estimator fitting.
            If set to "raise", the exception is raised. If a numeric value is given,
            FitFailedWarning is raised.

        Returns
        -------
        A dictionary of benchmark results for that clusterer
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

    def _run_validation(self, task: TaskObject, estimator: BaseClusterer):
        cv_splitter = task.cv_splitter
        scorers = task.scorers
        xy_dict = task.get_y_X("clustering")
        scores_df = evaluate(
            clusterer=estimator,
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
                scores[scorer.__name__] = row["test_" + scorer.__name__]
            scores["fit_time"] = row["fit_time"]
            # Collect prediction times from available columns
            for col in scores_df.columns:
                if col.endswith("_time") and col != "fit_time":
                    scores[col] = row[col]
            if "pred_time" not in scores:
                # Map internal_time or external_time to pred_time for consistency
                for time_col in ["internal_time", "external_time"]:
                    if time_col in row.index:
                        scores["pred_time"] = row[time_col]
                        break
            if self.return_data:
                y_test = row.get("y_test", None)
                y_pred = None
                for col in ["y_internal", "y_external"]:
                    if col in row.index:
                        y_pred = row[col]
                        break
                y_train = row.get("y_train", None)
                folds[ix] = FoldResults(scores, y_test, y_pred, y_train)
            else:
                folds[ix] = FoldResults(scores)
        return folds
