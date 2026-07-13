"""Benchmarking for regression estimators."""

__author__ = ["NAME-ASHWANIYADAV"]
__all__ = ["RegressionBenchmark"]

from sktime.benchmarking.classification import ClassificationBenchmark


class RegressionBenchmark(ClassificationBenchmark):
    """Regression benchmark.

    Run a series of regressors against a series of tasks defined via dataset
    loaders, cross validation splitting strategies and performance metrics,
    and return results as a df (as well as saving to file).

    Name alias for ``ClassificationBenchmark``, which is estimator type
    agnostic and also handles regressors. See ``ClassificationBenchmark``
    for the full parameter documentation.

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
        Valid keys depend on the value of ``backend``,
        see ``ClassificationBenchmark`` for details.

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
