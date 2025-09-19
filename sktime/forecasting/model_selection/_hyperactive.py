# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Placeholder record for hyperactive tuner."""

import numpy as np

from sktime.forecasting.base._delegate import _DelegatedForecaster
from sktime.utils.dependencies import _placeholder_record


@_placeholder_record("hyperactive.integrations.sktime", dependencies="hyperactive>=5")
class ForecastingOptCV(_DelegatedForecaster):
    """Tune an sktime forecaster via any optimizer in the hyperactive toolbox.

    ``ForecastingOptCV`` uses any available tuning engine from ``hyperactive``
    to tune a forecaster by backtesting.

    It passes backtesting results as scores to the tuning engine,
    which identifies the best hyperparameters.

    Any available tuning engine from hyperactive can be used, for example:

    * grid search - ``from hyperactive.opt import GridSearchSk as GridSearch``,
      this results in the same algorithm as ``ForecastingGridSearchCV``
    * hill climbing - ``from hyperactive.opt import HillClimbing``
    * optuna parzen-tree search - ``from hyperactive.opt.optuna import TPEOptimizer``

    Configuration of the tuning engine is as per the respective documentation.

    Formally, ``ForecastingOptCV`` does the following:

    In ``fit``:

    * wraps the ``forecaster``, ``scoring``, and other parameters
      into a ``SktimeForecastingExperiment`` instance, which is passed to the optimizer
      ``optimizer`` as the ``experiment`` argument.
    * Optimal parameters are then obtained from ``optimizer.solve``, and set
      as ``best_params_`` and ``best_forecaster_`` attributes.
    *  If ``refit=True``, ``best_forecaster_`` is fitted to the entire ``y`` and ``X``.

    In ``predict`` and ``predict``-like methods, calls the respective method
    of the ``best_forecaster_`` if ``refit=True``.

    Parameters
    ----------
    forecaster : sktime forecaster, BaseForecaster instance or interface compatible
        The forecaster to tune, must implement the sktime forecaster interface.

    optimizer : hyperactive BaseOptimizer
        The optimizer to be used for hyperparameter search.

    cv : sktime BaseSplitter descendant
        determines split of ``y`` and possibly ``X`` into test and train folds
        y is always split according to ``cv``, see above
        if ``cv_X`` is not passed, ``X`` splits are subset to ``loc`` equal to ``y``
        if ``cv_X`` is passed, ``X`` is split according to ``cv_X``

    strategy : {"refit", "update", "no-update_params"}, optional, default="refit"
        defines the ingestion mode when the forecaster sees new data when window expands
        "refit" = forecaster is refitted to each training window
        "update" = forecaster is updated with training window data, in sequence provided
        "no-update_params" = fit to first training window, re-used without fit or update

    update_behaviour : str, optional, default = "full_refit"
        one of {"full_refit", "inner_only", "no_update"}
        behaviour of the forecaster when calling update
        "full_refit" = both tuning parameters and inner estimator refit on all data seen
        "inner_only" = tuning parameters are not re-tuned, inner estimator is updated
        "no_update" = neither tuning parameters nor inner estimator are updated

    scoring : sktime metric (BaseMetric), str, or callable, optional (default=None)
        scoring metric to use in tuning the forecaster

        * sktime metric objects (BaseMetric) descendants can be searched
        with the ``registry.all_estimators`` search utility,
        for instance via ``all_estimators("metric", as_dataframe=True)``

        * If callable, must have signature
        ``(y_true: 1D np.ndarray, y_pred: 1D np.ndarray) -> float``,
        assuming np.ndarrays being of the same length, and lower being better.
        Metrics in sktime.performance_metrics.forecasting are all of this form.

        * If str, uses registry.resolve_alias to resolve to one of the above.
          Valid strings are valid registry.craft specs, which include
          string repr-s of any BaseMetric object, e.g., "MeanSquaredError()";
          and keys of registry.ALIAS_DICT referring to metrics.

        * If None, defaults to MeanAbsolutePercentageError()

    refit : bool, optional (default=True)
        True = refit the forecaster with the best parameters on the entire data in fit
        False = no refitting takes place. The forecaster cannot be used to predict.
        This is to be used to tune the hyperparameters, and then use the estimator
        as a parameter estimator, e.g., via get_fitted_params or PluginParamsForecaster.

    error_score : "raise" or numeric, default=np.nan
        Value to assign to the score if an exception occurs in estimator fitting. If set
        to "raise", the exception is raised. If a numeric value is given,
        FitFailedWarning is raised.

    cv_X : sktime BaseSplitter descendant, optional
        determines split of ``X`` into test and train folds
        default is ``X`` being split to identical ``loc`` indices as ``y``
        if passed, must have same number of splits as ``cv``

    backend : string, by default "None".
        Parallelization backend to use for runs.
        Runs parallel evaluate if specified and ``strategy="refit"``.

        - "None": executes loop sequentially, simple list comprehension
        - "loky", "multiprocessing" and "threading": uses ``joblib.Parallel`` loops
        - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``
        - "dask": uses ``dask``, requires ``dask`` package in environment
        - "dask_lazy": same as "dask",
          but changes the return to (lazy) ``dask.dataframe.DataFrame``.
        - "ray": uses ``ray``, requires ``ray`` package in environment

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

        - "ray": The following keys can be passed:

            - "ray_remote_args": dictionary of valid keys for ``ray.init``
            - "shutdown_ray": bool, default=True; False prevents ``ray`` from shutting
                down after parallelization.
            - "logger_name": str, default="ray"; name of the logger to use.
            - "mute_warnings": bool, default=False; if True, suppresses warnings

    Example
    -------
    Any available tuning engine from hyperactive can be used, for example:

    * grid search - ``from hyperactive.opt import GridSearchSk as GridSearch``
    * hill climbing - ``from hyperactive.opt import HillClimbing``
    * optuna parzen-tree search - ``from hyperactive.opt.optuna import TPEOptimizer``

    For illustration, we use grid search, this can be replaced by any other optimizer.

    1. defining the tuned estimator:
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.split import ExpandingWindowSplitter
    >>> from hyperactive.integrations.sktime import ForecastingOptCV
    >>> from hyperactive.opt import GridSearchSk as GridSearch
    >>>
    >>> param_grid = {"strategy": ["mean", "last", "drift"]}
    >>> tuned_naive = ForecastingOptCV(
    ...     NaiveForecaster(),
    ...     GridSearch(param_grid),
    ...     cv=ExpandingWindowSplitter(
    ...         initial_window=12, step_length=3, fh=range(1, 13)
    ...     ),
    ... )

    2. fitting the tuned estimator:
    >>> from sktime.datasets import load_airline
    >>> from sktime.split import temporal_train_test_split
    >>> y = load_airline()
    >>> y_train, y_test = temporal_train_test_split(y, test_size=12)
    >>>
    >>> tuned_naive.fit(y_train, fh=range(1, 13))
    ForecastingOptCV(...)
    >>> y_pred = tuned_naive.predict()

    3. obtaining best parameters and best estimator
    >>> best_params = tuned_naive.best_params_
    >>> best_estimator = tuned_naive.best_forecaster_
    """

    _tags = {
        "authors": "fkiraly",
        "maintainers": "fkiraly",
        "python_dependencies": "hyperactive>=5",
        # testing configuration
        # ---------------------
        "tests:vm": True,
    }

    def __init__(
        self,
        forecaster,
        optimizer,
        cv,
        strategy="refit",
        update_behaviour="full_refit",
        scoring=None,
        refit=True,
        error_score=np.nan,
        cv_X=None,
        backend=None,
        backend_params=None,
    ):
        self.forecaster = forecaster
        self.optimizer = optimizer
        self.cv = cv
        self.strategy = strategy
        self.update_behaviour = update_behaviour
        self.scoring = scoring
        self.refit = refit
        self.error_score = error_score
        self.cv_X = cv_X
        self.backend = backend
        self.backend_params = backend_params
        super().__init__()

    @classmethod
    def get_test_params(self, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return "default" set.

        Returns
        -------
        params : dict
            Parameters to create testing instance of the class.
            Instance will be created with ``estimator = ClassName(**params)``.
        """
        params1 = {"forecaster": None, "optimizer": None, "cv": None}
        return params1
