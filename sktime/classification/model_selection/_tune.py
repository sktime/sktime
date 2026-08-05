"""Tuning for time series classifiers."""

__author__ = ["fkiraly", "achieveordie", "atikulmunna"]

import time

import numpy as np
import pandas as pd

from sktime.classification._delegate import _DelegatedClassifier
from sktime.classification.model_evaluation._functions import _evaluate
from sktime.exceptions import NotFittedError
from sktime.utils.parallel import parallelize

# metric name suffixes indicating that lower values are better,
# used to determine ranking direction for plain sklearn metric callables,
# which do not carry a "lower_is_better" tag like sktime metric objects
_LOWER_IS_BETTER_SUFFIXES = ("_error", "_loss", "_deviance", "_risk")


def _metric_lower_is_better(metric):
    """Return True if lower values of the metric indicate better performance."""
    name = getattr(metric, "__name__", "").lower()
    return name.endswith(_LOWER_IS_BETTER_SUFFIXES)


def _check_scoring_callables(scoring, default):
    """Coerce scoring to a non-empty list of callables.

    Parameters
    ----------
    scoring : callable, list of callables, or None
        scoring as passed to the tuner
    default : callable
        default metric to use if scoring is None

    Returns
    -------
    list of callables, the metrics to evaluate

    Raises
    ------
    TypeError
        if scoring or an element of it is not callable
    """
    if scoring is None:
        scoring = [default]
    if not isinstance(scoring, list):
        scoring = [scoring]
    for metric in scoring:
        if not callable(metric):
            raise TypeError(
                "Error in grid search tuner: scoring must be a callable metric "
                "with signature (y_true, y_pred) -> float, a list of such "
                f"callables, or None, but found {metric!r}. String scoring "
                "values are not supported, pass the metric callable instead, "
                "e.g., accuracy_score from sklearn.metrics."
            )
    return scoring


def _resolve_cv(cv):
    """Resolve the cv parameter to a splitter with a split method.

    int and None are resolved to deterministic KFold, so that all
    parameter candidates are evaluated on identical splits.
    None resolves to 3 splits, same number as the evaluate default.
    """
    from sklearn.model_selection import check_cv

    if cv is None:
        cv = 3
    return check_cv(cv)


def _check_param_grid(param_grid):
    """Validate param_grid, from sklearn 1.0.2, before it was removed."""
    if hasattr(param_grid, "items"):
        param_grid = [param_grid]

    for p in param_grid:
        for name, v in p.items():
            if isinstance(v, np.ndarray) and v.ndim > 1:
                raise ValueError("Parameter array should be one-dimensional.")

            if isinstance(v, str) or not isinstance(v, (np.ndarray, list, tuple)):
                raise ValueError(
                    f"Parameter grid for parameter ({name}) needs to"
                    f" be a list or numpy array, but got ({type(v)})."
                    " Single values need to be wrapped in a list"
                    " with one element."
                )

            if len(v) == 0:
                raise ValueError(
                    f"Parameter values for parameter ({name}) need "
                    "to be a non-empty sequence."
                )


def _fit_and_score(params, meta):
    """Fit and score estimator with given parameters.

    Root level function for parallelization, called from
    _grid_search_fit, within parallelize.
    """
    estimator = meta["estimator"].clone()
    estimator.set_params(**params)

    out = _evaluate(
        estimator=estimator,
        cv=meta["cv"],
        X=meta["X"],
        y=meta["y"],
        scoring=meta["scoring"],
        error_score=meta["error_score"],
    )

    # keep scores and timings, aggregate means over folds
    out = out.filter(regex="^(test_|fit_time|pred)", axis=1)
    out = out.mean()
    out = out.add_prefix("mean_")

    out["params"] = params

    return out


def _grid_search_fit(tuner, X, y, default_scoring):
    """Run grid search backtesting for tuner, common to TSC and TSR tuners.

    Writes fitted attributes to tuner: cv_results_, best_index_, best_score_,
    best_params_, best_estimator_, n_splits_, scorer_, multimetric_,
    and refit_time_ if refit=True.

    Parameters
    ----------
    tuner : TSCGridSearchCV or TSRGridSearchCV instance
    X : panel data container, in an sktime compatible Panel mtype
    y : 2D np.ndarray, target values
    default_scoring : callable, metric to use if tuner.scoring is None
    """
    from sklearn.model_selection import ParameterGrid

    if y.ndim > 1 and y.shape[1] == 1:
        y = y.flatten()

    scoring = _check_scoring_callables(tuner.scoring, default_scoring)
    primary_metric = scoring[0]
    scoring_name = f"test_{primary_metric.__name__}"

    cv = _resolve_cv(tuner.cv)

    _check_param_grid(tuner.param_grid)
    candidate_params = list(ParameterGrid(tuner.param_grid))

    if tuner.verbose > 0:
        n_candidates = len(candidate_params)
        n_splits = cv.get_n_splits()
        print(  # noqa: T201
            f"Fitting {n_splits} folds for each of {n_candidates} candidates,"
            f" totalling {n_candidates * n_splits} fits"
        )

    # resolve parallelization backend, legacy n_jobs maps to joblib backends
    backend = tuner.backend
    backend_params = tuner.backend_params if tuner.backend_params else {}
    if backend is None and tuner.n_jobs is not None:
        backend = "loky"
        backend_params = {"n_jobs": tuner.n_jobs, **backend_params}

    meta = {
        "estimator": tuner.estimator,
        "X": X,
        "y": y,
        "cv": cv,
        "scoring": scoring,
        "error_score": tuner.error_score,
    }

    out = parallelize(
        fun=_fit_and_score,
        iter=candidate_params,
        meta=meta,
        backend=backend,
        backend_params=backend_params,
    )

    if len(out) < 1:
        raise ValueError(
            "No fits were performed. "
            "Was the CV iterator empty? "
            "Were there no candidates?"
        )

    results = pd.DataFrame(out)

    # rank results, according to whether greater is better for the given scoring,
    # with min method for ties, same as sklearn rank_test_score semantics
    lower_is_better = _metric_lower_is_better(primary_metric)
    results[f"rank_{scoring_name}"] = results.loc[:, f"mean_{scoring_name}"].rank(
        ascending=lower_is_better, method="min"
    )

    tuner.cv_results_ = results

    # select best parameters
    tuner.best_index_ = results.loc[:, f"rank_{scoring_name}"].argmin()
    # raise error if all fits in evaluate failed because all score values are NaN
    if tuner.best_index_ == -1:
        raise NotFittedError(
            f"""All fits of estimator failed,
            set error_score='raise' to see the exceptions.
            Failed estimator: {tuner.estimator}"""
        )
    tuner.best_score_ = results.loc[tuner.best_index_, f"mean_{scoring_name}"]
    tuner.best_params_ = results.loc[tuner.best_index_, "params"]
    tuner.best_estimator_ = tuner.estimator.clone().set_params(**tuner.best_params_)

    tuner.n_splits_ = cv.get_n_splits()
    tuner.scorer_ = primary_metric
    tuner.multimetric_ = len(scoring) > 1

    # refit model with best parameters
    if tuner.refit:
        refit_start_time = time.perf_counter()
        tuner.best_estimator_.fit(X=X, y=y)
        tuner.refit_time_ = time.perf_counter() - refit_start_time

    return tuner


class TSCGridSearchCV(_DelegatedClassifier):
    """Exhaustive search over specified parameter values for an estimator.

    Optimizes hyper-parameters of ``estimator`` by exhaustive grid search,
    using backtesting via ``sktime`` native ``evaluate`` for classifiers.

    In ``fit``, for each parameter combination in ``param_grid``,
    performance is backtested via
    ``sktime.classification.model_evaluation.evaluate``,
    on the data provided, using the cross-validation scheme ``cv``
    and the scoring metric ``scoring``.

    The best parameter combination, as per mean test score across folds,
    is then selected, and set as ``best_params_``.
    If ``refit=True``, ``best_estimator_`` is fitted to the entire data.

    In ``predict`` and ``predict``-like methods, calls the respective method
    of ``best_estimator_``, if ``refit=True``.

    Parameters
    ----------
    estimator : sktime classifier, BaseClassifier instance or interface compatible
        The classifier to tune, must implement the sktime classifier interface.

    param_grid : dict or list of dictionaries
        Dictionary with parameters names (``str``) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.

    scoring : callable, list of callables, or None, default=None
        Metric(s) to evaluate the performance of the cross-validated model on
        the test set, passed to ``evaluate`` internally.

        - a callable must have signature
          ``(y_true: 1D np.ndarray, y_pred: 1D np.ndarray) -> float``,
          e.g., ``accuracy_score`` from ``sklearn.metrics``
        - if a list of callables, the first metric in the list
          is used for ranking and selection of the best parameters,
          all metrics are computed and available in ``cv_results_``
        - if None, defaults to ``accuracy_score``

        Whether lower or higher values are better is determined from the
        metric name: names ending in ``_error``, ``_loss``, ``_deviance``,
        or ``_risk`` are assumed lower-is-better, all other higher-is-better.

    n_jobs : int, default=None
        Number of jobs to run in parallel for the grid search,
        via the ``loky`` backend of ``joblib``.
        ``None`` means 1, ``-1`` means using all processors.
        Retained for backward compatibility, ignored if ``backend`` is passed.
        For fine grained control of parallelization,
        use ``backend`` and ``backend_params`` instead.

    refit : bool, default=True
        Refit an estimator using the best found parameters on the whole
        dataset.
        If ``False``, ``predict`` and ``predict``-like methods will raise,
        and the tuner can be used only as a parameter estimator,
        e.g., via ``get_fitted_params``.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy, used in
        ``evaluate`` internally. Possible inputs for cv are:

        - None, to use the default 3-fold cross validation via ``KFold``,
        - integer, to specify the number of folds in a ``KFold`` splitter,
        - CV splitter with a ``split`` method,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, ``KFold`` is instantiated with
        ``shuffle=False``, so all parameter candidates are evaluated
        on identical, deterministic splits.

    verbose : int, default=0
        Controls the verbosity: the higher, the more messages.

    pre_dispatch : int, or str, default='2*n_jobs'
        Retained for backward compatibility, this parameter is ignored.
        Parallelization behaviour can be controlled via
        ``backend`` and ``backend_params``.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    return_train_score : bool, default=False
        Retained for backward compatibility, this parameter is ignored.
        Train scores are not computed by the native grid search.

    tune_by_variable : bool, optional (default=False)
        Whether to tune parameter by each time series variable separately,
        in case of multivariate data passed to the tuning estimator.
        Only applies if time series passed are strictly multivariate.
        If True, clones of the estimator will be fit to each variable separately,
        and are available in fields of the classifiers_ attribute.
        Has the same effect as applying ColumnEnsembleClassifier wrapper to self.
        If False, the same best parameter is selected for all variables.

    backend : {"dask", "loky", "multiprocessing", "threading"}, optional, default=None
        Runs parallel grid search over the parameter candidates, if specified.

        - "None": executes loop sequentially, simple list comprehension
        - "loky", "multiprocessing" and "threading": uses ``joblib.Parallel`` loops
        - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``
        - "dask": uses ``dask``, requires ``dask`` package in environment

        Recommendation: Use "dask" or "loky" for parallel grid search.

    backend_params : dict, optional
        additional parameters passed to the backend as config.
        Directly passed to ``utils.parallel.parallelize``.
        Valid keys depend on the value of ``backend``:

        - "None": no additional parameters, ``backend_params`` is ignored
        - "loky", "multiprocessing" and "threading": default ``joblib`` backends
          any valid keys for ``joblib.Parallel`` can be passed here, e.g., ``n_jobs``,
          with the exception of ``backend`` which is directly controlled by ``backend``.
        - "dask": any valid keys for ``dask.compute`` can be passed,
          e.g., ``scheduler``

    Attributes
    ----------
    cv_results_ : pd.DataFrame
        DataFrame with one row per parameter candidate, with columns:

        - ``mean_test_<metric_name>``: mean test score across folds,
          one column per metric in ``scoring``
        - ``mean_fit_time``: mean fit time across folds, in seconds
        - ``mean_pred_time``: mean prediction time across folds, in seconds
        - ``params``: the parameter settings of the candidate
        - ``rank_test_<metric_name>``: rank of the candidate, 1 is best,
          for the primary (first) metric in ``scoring``

    best_estimator_ : estimator
        Clone of ``estimator`` with the best found parameters set.
        Fitted to the entire data if ``refit=True``, otherwise unfitted.

    best_score_ : float
        Mean cross-validated test score of the best parameter candidate.

    best_params_ : dict
        Parameter setting that gave the best mean test score.

    best_index_ : int
        The row index in ``cv_results_`` corresponding to the
        best parameter candidate.

    scorer_ : callable
        The primary metric used to rank candidates and select best parameters.

    n_splits_ : int
        The number of cross-validation splits (folds).

    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.
        This is present only if ``refit=True``.

    multimetric_ : bool
        Whether multiple metrics were passed in ``scoring``.

    classes_ : ndarray of shape (n_classes,)
        The class labels.

    Examples
    --------
    >>> from sklearn.metrics import accuracy_score
    >>> from sklearn.model_selection import KFold
    >>> from sktime.classification.dummy import DummyClassifier
    >>> from sktime.classification.model_selection import TSCGridSearchCV
    >>> from sktime.datasets import load_unit_test
    >>> X, y = load_unit_test()
    >>> tuned_dummy = TSCGridSearchCV(
    ...     DummyClassifier(),
    ...     {"strategy": ["most_frequent", "stratified"]},
    ...     scoring=accuracy_score,
    ...     cv=KFold(n_splits=2, shuffle=False),
    ... )
    >>> tuned_dummy = tuned_dummy.fit(X, y)
    >>> y_pred = tuned_dummy.predict(X)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["fkiraly", "achieveordie", "atikulmunna"],
        # estimator type
        # --------------
        "X_inner_mtype": ["pd-multiindex", "nested_univ", "numpy3D"],
        "y_inner_mtype": ["numpy2D"],
        "capability:multivariate": True,
        "capability:multioutput": True,
        "capability:unequal_length": True,
        "capability:missing_values": True,
        "capability:multithreading": True,
        "capability:predict_proba": True,
        "capability:categorical_in_X": True,
        # CI and test flags
        # -----------------
        "tests:core": True,  # should tests be triggered by framework changes?
    }

    # attribute for _DelegatedClassifier, which then delegates
    #     all non-overridden methods to getattr(self, _delegate_name)
    #     see further details in _DelegatedClassifier docstring
    _delegate_name = "best_estimator_"

    def __init__(
        self,
        estimator,
        param_grid,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=False,
        tune_by_variable=False,
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
        self.backend = backend
        self.backend_params = backend_params

        super().__init__()

        if self.tune_by_variable:
            self.set_tags(**{"capability:multioutput": False})

    def _fit(self, X, y):
        """Fit time series classifier to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            if self.get_tag("X_inner_mtype") = "pd-multiindex":
            pd.DataFrame with columns = variables,
            index = pd.MultiIndex with first level = instance indices,
            second level = time indices
            for list of other mtypes, see datatypes.SCITYPE_REGISTER
            for specifications, see examples/AA_datatypes_and_datasets.ipynb
        y : guaranteed to be of a type in self.get_tag("y_inner_mtype")
            1D iterable, of shape [n_instances]
            or 2D iterable, of shape [n_instances, n_dimensions]
            class labels for fitting
            if self.get_tag("capability:multioutput") = False, guaranteed to be 1D
            if self.get_tag("capability:multioutput") = True, guaranteed to be 2D

        Returns
        -------
        self : Reference to self.
        """
        from sklearn.metrics import accuracy_score

        return _grid_search_fit(self, X=X, y=y, default_scoring=accuracy_score)

    def _check_refit_for_predict(self, method_name):
        """Raise error if refit=False and a predict-like method is called."""
        if not self.refit:
            raise RuntimeError(
                f"In {self.__class__.__name__}, refit must be True to make"
                f" predictions, but found refit=False. If refit=False,"
                f" {self.__class__.__name__} can be used only to tune"
                " hyper-parameters, as a parameter estimator."
            )

    def _predict(self, X):
        """Predict labels for sequences in X.

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
        y : predictions of labels for X, np.ndarray
        """
        self._check_refit_for_predict("predict")
        estimator = self._get_delegate()
        y_pred = estimator.predict(X=X)
        # coerce to 2D np.ndarray, the return contract for y_inner_mtype numpy2D
        if hasattr(y_pred, "to_numpy"):
            y_pred = y_pred.to_numpy()
        y_pred = np.asarray(y_pred)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        return y_pred

    def _predict_proba(self, X):
        """Predict labels probabilities for sequences in X.

        private _predict_proba containing the core logic, called from predict_proba

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")

        Returns
        -------
        y : 2D array of shape [n_instances, n_classes] - predicted class probabilities
        """
        self._check_refit_for_predict("predict_proba")
        return super()._predict_proba(X)

    def _get_fitted_params(self):
        """Get fitted parameters.

        private _get_fitted_params, called from get_fitted_params

        State required:
            Requires state to be "fitted".

        Returns
        -------
        fitted_params : dict with str keys
            A dict containing the best hyper parameters and the parameters of
            the best estimator (if available), merged together with the former
            taking precedence.
        """
        fitted_params = {}
        try:
            fitted_params = self.best_estimator_.get_fitted_params()
        except (NotFittedError, NotImplementedError):
            pass
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
            special parameters are defined for a value, will return ``"default"`` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``.
        """
        from sklearn.gaussian_process.kernels import RBF, DotProduct
        from sklearn.metrics import accuracy_score

        from sktime.classification.kernel_based import TimeSeriesSVC
        from sktime.dists_kernels import AggrDist

        mean_eucl_tskernel = AggrDist(DotProduct())
        mean_rbf_tskernel = AggrDist(RBF())

        param1 = {
            "estimator": TimeSeriesSVC(kernel=mean_rbf_tskernel, probability=True),
            "param_grid": {"C": [0.1, 1]},
        }

        param2 = {
            "estimator": TimeSeriesSVC(kernel=mean_eucl_tskernel, probability=True),
            "param_grid": {"kernel__transformer": [DotProduct(), RBF()]},
            "scoring": accuracy_score,
        }

        return [param1, param2]
