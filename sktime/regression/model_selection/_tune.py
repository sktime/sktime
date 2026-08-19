"""Tuning for time series regressors."""

__author__ = ["ksharma6", "yash-sangwan"]

import numpy as np

from sktime.regression._delegate import _DelegatedRegressor


# todo 1.3.0: remove the n_jobs and pre_dispatch parameters, from the
# signature and the docstring, and remove the call to
# _resolve_deprecated_parallel in _fit_tuner
class TSRGridSearchCV(_DelegatedRegressor):
    """Exhaustive search over specified parameter values for a regressor.

    Optimizes hyper-parameters of ``estimator`` by exhaustive grid search, using
    ``sktime`` native backtesting via ``regression.model_evaluation.evaluate``.

    In ``fit``, each parameter combination in ``param_grid`` is backtested on the
    data passed, using the cross-validation scheme ``cv`` and the metric
    ``scoring``. All candidates are evaluated on identical folds.

    The parameter combination with the best mean test score is set as
    ``best_params_``, and a clone of ``estimator`` with those parameters is set as
    ``best_estimator_``. If ``refit`` is not False, ``best_estimator_`` is fitted
    to the entire data, and ``predict`` of the tuner calls ``predict`` of
    ``best_estimator_``.

    Parameters
    ----------
    estimator : sktime regressor, BaseRegressor instance or interface compatible
        The regressor to tune, must implement the sktime regressor interface.

    param_grid : dict or list of dictionaries
        Dictionary with parameters names (``str``) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.

    scoring : None, str, callable, sklearn scorer, or list or dict of these
        Metric or metrics to evaluate the cross-validated model with.

        - a callable must have signature ``(y_true, y_pred) -> float``, e.g.,
          ``r2_score`` from ``sklearn.metrics``. Its value is reported as is
        - a string must name a scikit-learn scorer, e.g., ``"r2"``. Values are
          reported with the sign convention of the scorer, so values of
          ``"neg_mean_squared_error"`` are negative
        - a list or dict selects multiple metrics. The first is used to rank
          candidates, unless ``refit`` names another. Dict keys are used as the
          metric names in ``cv_results_``
        - if None, defaults to ``r2_score``

    n_jobs : int, optional, default="deprecated"
        Number of jobs to run in parallel over the parameter candidates.

        Deprecated, and will be removed in sktime 1.3.0. If passed, the value is
        written to ``backend_params``, and ``backend`` defaults to ``"loky"``,
        so behaviour is unchanged. To retain the behaviour after removal, pass
        ``backend="loky"`` and ``backend_params={"n_jobs": ...}`` instead.

    refit : bool, str, or callable, default=True
        Refit ``best_estimator_`` using the best found parameters on the whole
        dataset. If False, ``predict`` raises, and the tuner can be used only to
        tune hyper-parameters, e.g., as a parameter estimator via
        ``get_fitted_params``.

        For multi-metric evaluation, this can be a ``str`` naming the metric to
        select the best parameters by.

        Where there are considerations other than the best score in choosing the
        best parameters, ``refit`` can be a callable, which is applied to
        ``cv_results_`` and returns the selected ``best_index_``. In that case
        ``best_score_`` is not available.

    cv : int, cross-validation generator, iterable of splits, or None, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds in a ``KFold``,
        - a cross-validation splitter with a ``split`` method,
        - an iterable yielding (train, test) splits as arrays of indices.

        For integer and None inputs, ``KFold`` is used, instantiated with
        ``shuffle=False``, so the splits are the same across calls.

        Splits are computed once, before the search, so that all parameter
        candidates are evaluated on the same folds.

    verbose : int, default=0
        Controls the verbosity. If positive, the number of fits is printed.

    pre_dispatch : int or str, optional, default="deprecated"
        Number of jobs dispatched during parallel execution, a ``joblib``
        parameter.

        Deprecated, and will be removed in sktime 1.3.0. If passed, the value is
        written to ``backend_params``, and ``backend`` defaults to ``"loky"``,
        so behaviour is unchanged. To retain the behaviour after removal, pass
        ``backend="loky"`` and ``backend_params={"pre_dispatch": ...}`` instead.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    return_train_score : bool, default=False
        Retained for backwards compatibility, this parameter is ignored.
        Train scores are not computed, ``cv_results_`` holds test scores only.

    tune_by_variable : bool, optional (default=False)
        Whether to tune parameter by each time series variable separately,
        in case of multivariate data passed to the tuning estimator.
        Only applies if time series passed are strictly multivariate.
        If True, clones of the estimator will be fit to each variable separately,
        and are available in fields of the regressors_ attribute.
        Has the same effect as applying ColumnEnsembleRegressor wrapper to self.
        If False, the same best parameter is selected for all variables.

    greater_is_better : "auto", bool, optional, default="auto"
        Whether higher values of the reported metric are better, used to rank
        the parameter candidates.

        - "auto" determines the direction from the metric. Scikit-learn scorers
          are higher-is-better by their sign convention. For metric callables,
          the direction is inferred from the metric, e.g., ``r2_score`` is
          higher-is-better, and ``mean_squared_error`` is lower-is-better
        - True or False set the direction explicitly, for all metrics

    backend : str, optional, default=None
        Parallelization backend for the search over parameter candidates.

        - None: executes loop sequentially, simple list comprehension
        - "loky", "multiprocessing" and "threading": uses ``joblib.Parallel`` loops
        - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``
        - "dask": uses ``dask``, requires ``dask`` package in environment
        - "ray": uses ``ray``, requires ``ray`` package in environment

        Recommendation: use "dask" or "loky" for parallel grid search.
        "threading" is unlikely to see speed ups due to the GIL.

    backend_params : dict, optional
        Additional parameters passed to the backend as config, directly passed
        to ``utils.parallel.parallelize``. Valid keys depend on ``backend``,
        see there for details.

    Attributes
    ----------
    cv_results_ : dict of str to numpy (masked) ndarray
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``. Contains ``params``, one
        ``param_<name>`` column per searched parameter, fit and score timings,
        and per metric the per-fold scores ``split<i>_test_<name>``, their mean
        ``mean_test_<name>``, standard deviation ``std_test_<name>``, and rank
        ``rank_test_<name>``, 1 being the best.

        For a single metric, ``<name>`` is ``score``, e.g., ``mean_test_score``.
        For multiple metrics, ``<name>`` is the name of the respective metric.

    best_estimator_ : estimator
        Clone of ``estimator`` with the best found parameters set.
        Fitted to the entire data if ``refit`` is not False, otherwise unfitted.

    best_score_ : float
        Mean cross-validated score of ``best_estimator_``.
        Not available if ``refit`` is a callable.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

    best_index_ : int
        The index in the ``cv_results_`` arrays which corresponds to the best
        candidate parameter setting.

    scorer_ : callable or dict of callable
        Metric used on the held out data to choose the best parameters.
        For multi-metric evaluation, a dict of metric name to metric.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.
        This is present only if ``refit`` is not False.

    multimetric_ : bool
        Whether multiple metrics were passed in ``scoring``.

    See Also
    --------
    ParameterGrid : Generates all the combinations of a hyperparameter grid.
    sktime.regression.model_evaluation.evaluate : Backtesting used internally.

    Examples
    --------
    >>> from sktime.datasets import load_unit_test
    >>> from sktime.regression.dummy import DummyRegressor
    >>> from sktime.regression.model_selection import TSRGridSearchCV
    >>>
    >>> X, y = load_unit_test(split="train")
    >>> tuned = TSRGridSearchCV(
    ...     DummyRegressor(),
    ...     param_grid={"strategy": ["mean", "median"]},
    ...     cv=2,
    ... )
    >>> tuned = tuned.fit(X, y.astype("float"))
    >>> y_pred = tuned.predict(X)
    >>> best_params = tuned.best_params_
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["ksharma6", "yash-sangwan"],
        # estimator type
        # --------------
        "X_inner_mtype": "pd-multiindex",
        "y_inner_mtype": ["numpy2D"],
        "capability:multivariate": True,
        "capability:multioutput": True,
        "capability:unequal_length": True,
        "capability:missing_values": True,
        "capability:multithreading": True,
        "capability:categorical_in_X": True,
    }

    # attribute for _DelegatedRegressor, which then delegates
    #     all non-overridden methods are same as of getattr(self, _delegate_name)
    #     see further details in _DelegatedRegressor docstring
    _delegate_name = "best_estimator_"

    def __init__(
        self,
        estimator,
        param_grid,
        scoring=None,
        n_jobs="deprecated",
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="deprecated",
        error_score=np.nan,
        return_train_score=False,
        tune_by_variable=False,
        greater_is_better="auto",
        backend=None,
        backend_params=None,
    ):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.cv = cv
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self.return_train_score = return_train_score
        self.tune_by_variable = tune_by_variable
        self.greater_is_better = greater_is_better
        self.backend = backend
        self.backend_params = backend_params

        super().__init__()

        if self.tune_by_variable:
            self.set_tags(**{"capability:multioutput": False})

    def _fit(self, X, y):
        """Fit time series regressor to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            pd.DataFrame with columns = variables,
            index = pd.MultiIndex with first level = instance indices,
            second level = time indices
        y : guaranteed to be of a type in self.get_tag("y_inner_mtype")
            2D np.ndarray of shape [n_instances, n_outputs], target values

        Returns
        -------
        self : Reference to self.
        """
        # deferred import, sibling type modules must not cross-import at module
        # level, see sktime/tests/test_cross_module_imports.py
        from sktime.classification.model_selection._tune import _fit_tuner

        return _fit_tuner(self, X, y, estimator_type="regressor")

    def _predict(self, X):
        """Predict target values for sequences in X.

        private _predict containing the core logic, called from predict

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")

        Returns
        -------
        y : 2D np.ndarray of shape [n_instances, n_outputs], predicted values
        """
        from sktime.classification.model_selection._tune import (
            _check_refit_for_predict,
            _coerce_prediction,
        )

        _check_refit_for_predict(self)
        return _coerce_prediction(self._get_delegate().predict(X=X))

    def _get_fitted_params(self):
        """Get fitted parameters.

        private _get_fitted_params, called from get_fitted_params

        State required:
            Requires state to be "fitted".

        Returns
        -------
        fitted_params : dict with str keys
            The best hyper-parameters, and the fitted parameters of
            ``best_estimator_`` if available, the former taking precedence.
        """
        fitted_params = {}
        # best_estimator_ is fitted only if refit is not False
        if self.refit:
            fitted_params = self.best_estimator_.get_fitted_params()

        fitted_params = {**fitted_params, **self.best_params_}
        fitted_params.update(self._get_fitted_params_default())

        return fitted_params

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            For regressors, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        from sklearn.metrics import r2_score

        from sktime.dists_kernels import AggrDist, ScipyDist
        from sktime.regression.distance_based import KNeighborsTimeSeriesRegressor

        mean_eucl_tsdist = AggrDist(ScipyDist(metric="euclidean"))
        mean_cb_tsdist = AggrDist(ScipyDist(metric="cityblock"))

        param1 = {
            "estimator": KNeighborsTimeSeriesRegressor(distance=mean_eucl_tsdist),
            "param_grid": {"n_neighbors": [1, 3, 5]},
        }

        param2 = {
            "estimator": KNeighborsTimeSeriesRegressor(distance=mean_cb_tsdist),
            "param_grid": {"distance__metric": ["euclidean", "cityblock"]},
            "scoring": r2_score,
        }

        return [param1, param2]
