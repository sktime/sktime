# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Random search forecaster."""

import numpy as np
from sklearn.model_selection import ParameterSampler

from sktime.forecasting.model_selection._base import BaseGridSearch


class ForecastingRandomizedSearchCV(BaseGridSearch):
    """Perform randomized-search cross-validation to find optimal model parameters.

    The forecaster is fit on the initial window and then temporal
    cross-validation is used to find the optimal parameter

    Randomized cross-validation is performed based on a cross-validation
    iterator encoding the cross-validation scheme, the parameter distributions to
    search over, and (optionally) the evaluation metric for comparing model
    performance. As in scikit-learn, tuning works through the common
    hyper-parameter interface which allows to repeatedly fit and evaluate
    the same forecaster with different hyper-parameters.

    Parameters
    ----------
    forecaster : sktime forecaster, BaseForecaster instance or interface compatible
        The forecaster to tune, must implement the sktime forecaster interface.
        sklearn regressors can be used, but must first be converted to forecasters
        via one of the reduction compositors, e.g., via ``make_reduction``

    cv : sktime time series splitter
        Re-sampling strategy for cross-validation, must be an instance of a sktime
        time series splitter, e.g. ``SlidingWindowSplitter()``

    param_distributions : dict or list of dicts
        Dictionary with parameters names (``str``) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).

        * If a list is given, it is sampled uniformly.
        * If a list of dicts is given, first a dict is sampled uniformly, and
          then a parameter is sampled using that dict as above.

    n_iter : int, default=10
        Number of parameter settings that are sampled. n_iter trades
        off runtime vs quality of the solution.

    scoring : sktime metric (BaseMetric), str, or callable, optional (default=None)
        scoring metric to use in tuning the forecaster

        * sktime metric objects (BaseMetric) descendants can be searched
          with the ``registry.all_estimators`` search utility,
          for instance via ``all_estimators("metric", as_dataframe=True)``

        * If callable, must have signature
          ``(y_true: 1D np.ndarray, y_pred: 1D np.ndarray) -> float``,
          with ``np.ndarray`` being of the same length, and lower being better.

        * If str, uses registry.resolve_alias to resolve to one of the above.
          Valid strings are valid registry.craft specs, which include
          string repr-s of any BaseMetric object, e.g., "MeanSquaredError()";
          and keys of registry.ALIAS_DICT referring to metrics.

        * If None, defaults to ``MeanAbsolutePercentageError()``

    strategy : {"refit", "update", "no-update_params"}, optional, default="refit"
        data ingestion strategy in fitting cv, passed to ``evaluate`` internally
        defines the ingestion mode when the forecaster sees new data when window expands

        * ``"refit"`` = a new copy of the forecaster is fitted to each training window
        * ``"update"`` = forecaster is updated with training window data,
          in sequence provided
        * ``"no-update_params"`` = fit to first training window,
          re-used without fit or update

    update_behaviour : str, optional, default = "full_refit"
        one of {"full_refit", "inner_only", "no_update"}
        behaviour of the forecaster when calling update

        * ``"full_refit"`` = both tuning parameters and inner estimator refit on
          all data seen
        * ``"inner_only"`` = tuning parameters are not re-tuned, inner estimator is
          updated
        * ``"no_update"`` = neither tuning parameters nor inner estimator are updated

    refit : bool, optional (default=True)
        Whether to refit the forecaster with the best parameters on the entire data.

        * True = refit the forecaster with the best parameters
          on the entire data in ``fit``
        * False = no refitting takes place. The forecaster cannot be used to predict.
          This is to be used to tune the hyperparameters, and then use the estimator
          as a parameter estimator, e.g.,
          via ``get_fitted_params`` or ``PluginParamsForecaster``.

    tune_by_instance : bool, optional (default=False)
        Whether to tune parameter by each time series instance separately,
        in case of ``Panel`` or ``Hierarchical`` data passed to the tuning estimator.
        Only applies if time series passed are ``Panel`` or ``Hierarchical``.

        * If ``True``, clones of the forecaster will be fit to each instance separately,
          and are available in fields of the forecasters_ attribute.
          Has the same effect as applying ``ForecastByLevel`` wrapper to self.
        * If ``False``, the same best parameter is selected for all instances.

    tune_by_variable : bool, optional (default=False)
        Whether to tune parameter by each time series variable separately,
        in case of multivariate data passed to the tuning estimator.
        Only applies if time series passed are strictly multivariate.

        * If ``True``, clones of the forecaster will be fit to each variable separately,
          and are available in fields of the forecasters_ attribute.
          Has the same effect as applying ``ColumnEnsembleForecaster`` wrapper to self.
        * If ``False``, the same best parameter is selected for all variables.

    verbose: int, optional (default=0)
        Verbosity level. The higher, the more messages.

    return_n_best_forecasters: int, default=1
        In case the n best forecaster should be returned, this value can be set
        and the n best forecasters will be assigned to n_best_forecasters_.
        Set return_n_best_forecasters to -1 to return all forecasters.

    error_score : "raise" or numeric, default=np.nan
        Value to assign to the score if an exception occurs in estimator fitting. If set
        to "raise", the exception is raised. If a numeric value is given,
        FitFailedWarning is raised.

    backend : {"dask", "loky", "multiprocessing", "threading"}, by default "loky".
        Runs parallel evaluate if specified and ``strategy`` is set as "refit".

        - "None": executes loop sequentally, simple list comprehension
        - "loky", "multiprocessing" and "threading": uses ``joblib.Parallel`` loops
        - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``
        - "dask": uses ``dask``, requires ``dask`` package in environment
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
        - "dask": any valid keys for ``dask.compute`` can be passed, e.g., ``scheduler``

        - "ray": The following keys can be passed:

            - "ray_remote_args": dictionary of valid keys for ``ray.init``
            - "shutdown_ray": bool, default=True; False prevents ``ray`` from shutting
                down after parallelization.
            - "logger_name": str, default="ray"; name of the logger to use.
            - "mute_warnings": bool, default=False; if True, suppresses warnings

    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        Pass an int for reproducible output across multiple
        function calls.

    Attributes
    ----------
    best_index_ : int
    best_score_: float
        Score of the best model
    best_params_ : dict
        Best parameter values across the parameter grid
    best_forecaster_ : estimator
        Fitted estimator with the best parameters
    cv_results_ : dict
        Results from grid search cross validation
    n_best_forecasters_: list of tuples ("rank", <forecaster>)
        The "rank" is in relation to best_forecaster_
    n_best_scores_: list of float
        The scores of n_best_forecasters_ sorted from best to worst
        score of forecasters
    forecasters_ : pd.DataFramee
        DataFrame with all fitted forecasters and their parameters.
        Only present if tune_by_instance=True or tune_by_variable=True,
        and at least one of the two is applicable.
        In this case, the other attributes are not present in self,
        only in the fields of forecasters_.

    Examples
    --------
    >>> from sktime.datasets import load_shampoo_sales
    >>> from sktime.forecasting.model_selection import ForecastingRandomizedSearchCV
    >>> from sktime.split import ExpandingWindowSplitter
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> y = load_shampoo_sales()
    >>> fh = [1, 2, 3]
    >>> cv = ExpandingWindowSplitter(fh=fh)
    >>> forecaster = NaiveForecaster()
    >>> param_distributions = {"strategy": ["last", "mean", "drift"]}
    >>> rscv = ForecastingRandomizedSearchCV(
    ...     forecaster=forecaster,
    ...     param_distributions=param_distributions,
    ...     cv=cv,
    ...     n_iter=3,
    ...     random_state=42)
    >>> rscv.fit(y)
    ForecastingRandomizedSearchCV(...)
    >>> y_pred = rscv.predict(fh)

    Advanced randomized search with a ``scipy.stats`` distribution for a
    continuous hyperparameter, on a pipeline forecaster:

    >>> from scipy.stats import randint
    >>> from sktime.datasets import load_shampoo_sales
    >>> from sktime.forecasting.compose import TransformedTargetForecaster
    >>> from sktime.forecasting.model_selection import ForecastingRandomizedSearchCV
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.split import ExpandingWindowSplitter
    >>> from sktime.transformations.series.detrend import Detrender
    >>> y = load_shampoo_sales()
    >>> pipe = TransformedTargetForecaster(steps=[
    ...     ("detrender", Detrender()),
    ...     ("forecaster", NaiveForecaster(strategy="mean"))])
    >>> param_distributions = {
    ...     "forecaster__window_length": randint(2, 12),
    ...     "forecaster__strategy": ["mean", "last", "drift"],
    ... }
    >>> cv = ExpandingWindowSplitter(
    ...     initial_window=18, step_length=6, fh=[1, 2, 3])
    >>> rscv = ForecastingRandomizedSearchCV(
    ...     forecaster=pipe,
    ...     param_distributions=param_distributions,
    ...     cv=cv,
    ...     n_iter=5,
    ...     random_state=42)
    >>> rscv.fit(y)
    ForecastingRandomizedSearchCV(...)
    >>> y_pred = rscv.predict(fh=[1, 2, 3])
    """

    _tags = {
        "capability:random_state": True,
        "property:randomness": "derandomized",
        # CI and test flags
        # -----------------
        "tests:libs": ["sktime.forecasting.model_selection._base"],
    }

    def __init__(
        self,
        forecaster,
        cv,
        param_distributions,
        n_iter=10,
        scoring=None,
        strategy="refit",
        update_behaviour="full_refit",
        refit=True,
        tune_by_instance=False,
        tune_by_variable=False,
        verbose=0,
        return_n_best_forecasters=1,
        error_score=np.nan,
        backend="loky",
        backend_params=None,
        random_state=None,
        n_jobs="deprecated",
    ):
        super().__init__(
            forecaster=forecaster,
            scoring=scoring,
            strategy=strategy,
            refit=refit,
            cv=cv,
            verbose=verbose,
            return_n_best_forecasters=return_n_best_forecasters,
            backend=backend,
            update_behaviour=update_behaviour,
            error_score=error_score,
            tune_by_instance=tune_by_instance,
            tune_by_variable=tune_by_variable,
            backend_params=backend_params,
            n_jobs=n_jobs,
        )
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state

    def _run_search(self, evaluate_candidates):
        """Search n_iter candidates from param_distributions."""
        return evaluate_candidates(
            ParameterSampler(
                self.param_distributions, self.n_iter, random_state=self.random_state
            )
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict
        """
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.forecasting.trend import PolynomialTrendForecaster
        from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError
        from sktime.split import SingleWindowSplitter

        params = {
            "forecaster": NaiveForecaster(strategy="mean"),
            "cv": SingleWindowSplitter(fh=1),
            "param_distributions": {"window_length": [2, 5]},
            "scoring": MeanAbsolutePercentageError(symmetric=True),
        }

        params2 = {
            "forecaster": PolynomialTrendForecaster(),
            "cv": SingleWindowSplitter(fh=1),
            "param_distributions": {"degree": [1, 2]},
            "scoring": MeanAbsolutePercentageError(symmetric=True),
            "update_behaviour": "inner_only",
        }
        params3 = {
            "forecaster": NaiveForecaster(strategy="mean"),
            "cv": SingleWindowSplitter(fh=1),
            "param_distributions": {"window_length": [3, 4]},
            "scoring": "MeanAbsolutePercentageError(symmetric=True)",
            "update_behaviour": "no_update",
        }

        return [params, params2, params3]
