"""Benchmarking for regression estimators."""

__author__ = ["NAME-ASHWANIYADAV"]
__all__ = ["RegressionBenchmark"]

from sktime.benchmarking.classification import ClassificationBenchmark


class RegressionBenchmark(ClassificationBenchmark):
    """Regression benchmark.

    Run a series of regressors against a series of tasks defined via dataset
    loaders, cross validation splitting strategies and performance metrics,
    and return results as a df (as well as saving to file).

    Parameters
    ----------
    id_format: str, optional (default=None)
        A regex used to enforce task/estimator ID to match a certain format

    backend : string, by default "None".
        Parallelization backend to use for runs.
        See ``ClassificationBenchmark`` for the list of valid backends.

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
          Most common parameters:

            - "n_jobs": int, default=-1; number of jobs to run in parallel.
            - "verbose": int, default=0; verbosity level.
            - "timeout": float or None, default=None; timeout limit for each task.

          For full reference, see ``joblib.Parallel`` documentation:
          https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html
        - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``.
          any valid keys for ``joblib.Parallel`` can be passed here, e.g., ``n_jobs``,
          ``backend`` must be passed as a key of ``backend_params`` in this case.
          If ``n_jobs`` is not passed, it will default to ``-1``, other parameters
          will default to ``joblib`` defaults.
          See ``joblib.Parallel`` documentation above for common parameters.
        - "dask": any valid keys for ``dask.compute`` can be passed, e.g.,
          ``scheduler``. Most common parameters:

            - "scheduler": str, optional; scheduler to use, e.g., "threads",
              "processes", or "single-threaded".
            - "num_workers": int, optional; number of workers to use.
            - "optimize_graph": bool, default=True; whether to optimize the task
              graph before computation.

          For full reference, see ``dask.compute`` documentation:
          https://docs.dask.org/en/stable/generated/dask.compute.html
        - "ray": The following keys can be passed:

            - "ray_remote_args": dict, optional; valid keys for ``ray.init``, e.g.,
              ``"num_cpus"`` (int), ``"num_gpus"`` (int), ``"address"`` (str).
              For full reference, see ``ray.init`` documentation:
              https://docs.ray.io/en/latest/ray-core/api/doc/ray.init.html
            - "shutdown_ray": bool, default=True; False prevents ``ray`` from shutting
              down after parallelization.
            - "logger_name": str, default="ray"; name of the logger to use.
            - "mute_warnings": bool, default=False; if True, suppresses warnings

    return_data : bool, optional (default=False)
        Whether to return the prediction and the ground truth data in the results.

    Examples
    --------
    >>> from sklearn.metrics import mean_squared_error
    >>> from sklearn.model_selection import KFold
    >>> from sktime.benchmarking.regression import RegressionBenchmark
    >>> from sktime.regression.dummy import DummyRegressor
    >>> from sktime.utils._testing.panel import make_regression_problem
    >>> benchmark = RegressionBenchmark()
    >>> benchmark.add_estimator(DummyRegressor())
    >>> benchmark.add_task(
    ...     make_regression_problem,
    ...     KFold(n_splits=3),
    ...     [mean_squared_error],
    ... )
    >>> results_df = benchmark.run(None)  # doctest: +SKIP
    """
