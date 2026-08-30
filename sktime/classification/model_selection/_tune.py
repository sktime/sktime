"""Tuning for time series classifiers.

Also contains the grid search engine shared by the classification and regression
tuners. The regression tuner imports it with a function level import, as sibling
type modules must not cross-import at module level, see
``sktime/tests/test_cross_module_imports.py``.
"""

__author__ = ["fkiraly", "achieveordie", "yash-sangwan"]

import time

import numpy as np
from sklearn.model_selection import ParameterGrid

from sktime.classification._delegate import _DelegatedClassifier
from sktime.classification.model_evaluation import evaluate
from sktime.exceptions import NotFittedError
from sktime.utils.parallel import parallelize
from sktime.utils.sklearn._model_selection import _check_param_grid
from sktime.utils.sklearn._scoring import _resolve_scoring
from sktime.utils.warnings import warn


class _FixedSplitter:
    """Cross-validation splitter with pre-computed splits.

    Ensures that all parameter candidates are backtested on identical folds,
    also if the splitter passed by the user shuffles without a random state.

    Parameters
    ----------
    splits : list of pairs of 1D np.ndarray
        the (train, test) index pairs to yield from ``split``
    """

    def __init__(self, splits):
        self.splits = splits

    def split(self, X=None, y=None, groups=None):
        """Yield the pre-computed (train, test) splits."""
        return iter(self.splits)

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of splits."""
        return len(self.splits)


def _coerce_y(y):
    """Coerce y to 1D np.ndarray if it has a single output, else leave 2D."""
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.flatten()
    return y


def _resolve_cv(cv, y, estimator_type):
    """Resolve cv to a splitter with fixed splits.

    ``int`` and ``None`` are resolved as in sklearn ``check_cv``, i.e., to
    ``StratifiedKFold`` for classifiers with binary or multiclass ``y``,
    and to ``KFold`` otherwise, both with ``shuffle=False``.

    Splits are materialised once, so that all parameter candidates
    are backtested on the same folds.

    Parameters
    ----------
    cv : int, cross-validation generator, iterable of splits, or None
    y : 1D or 2D np.ndarray, target values, used for stratification
    estimator_type : str, one of "classifier" or "regressor"

    Returns
    -------
    _FixedSplitter, with the materialised splits
    """
    from sklearn.model_selection import check_cv

    cv = check_cv(cv, y, classifier=estimator_type == "classifier")

    instance_idx = np.arange(len(y))
    try:
        splits = list(cv.split(instance_idx, y))
    except TypeError:  # splitters that do not accept y
        splits = list(cv.split(instance_idx))

    if len(splits) == 0:
        raise ValueError(
            "Error in grid search tuner, the cross-validation splitter "
            f"{cv} produced no splits."
        )

    return _FixedSplitter(splits)


def _score_times(results):
    """Sum the prediction times of an evaluate result, per fold."""
    cols = [c for c in results.columns if c.endswith("_time") and c != "fit_time"]
    if len(cols) == 0:
        return np.zeros(len(results), dtype=float)
    return results[cols].sum(axis=1).to_numpy(dtype=float)


def _fit_and_score(params, meta):
    """Backtest one parameter candidate, via native evaluate.

    Root level function for parallelization, called from
    ``_run_grid_search``, within ``parallelize``.
    """
    estimator = meta["estimator"].clone().set_params(**params)
    scoring = meta["scoring"]

    # scoring is always passed explicitly, so the default of evaluate never applies,
    # and the same call is correct for classifiers and regressors alike
    results = evaluate(
        estimator,
        cv=meta["cv"],
        X=meta["X"],
        y=meta["y"],
        scoring=[entry.metric for entry in scoring],
        error_score=meta["error_score"],
    )

    scores = {}
    for entry in scoring:
        fold_scores = results[f"test_{entry.metric.__name__}"].to_numpy(dtype=float)
        scores[entry.name] = entry.sign * fold_scores

    return {
        "params": params,
        "scores": scores,
        "fit_time": results["fit_time"].to_numpy(dtype=float),
        "score_time": _score_times(results),
    }


def _param_columns(params):
    """Build the ``param_<name>`` columns of cv_results_, as masked arrays."""
    names = sorted({name for candidate in params for name in candidate})

    columns = {}
    for name in names:
        column = np.ma.MaskedArray(np.empty(len(params)), mask=True, dtype=object)
        for i, candidate in enumerate(params):
            if name in candidate:
                column[i] = candidate[name]
        columns[f"param_{name}"] = column
    return columns


def _rank_scores(mean_scores, greater_is_better):
    """Rank mean scores, 1 is best, ties get the minimum rank, NaN ranks last."""
    from scipy.stats import rankdata

    scores = np.asarray(mean_scores, dtype=float)
    worst = -np.inf if greater_is_better else np.inf
    scores = np.where(np.isnan(scores), worst, scores)
    if greater_is_better:
        scores = -scores
    return np.asarray(rankdata(scores, method="min"), dtype=np.int32)


def _make_cv_results(candidates, scoring, multimetric):
    """Assemble sklearn style cv_results_ from the candidate backtesting results.

    Parameters
    ----------
    candidates : list of dict, the returns of ``_fit_and_score``, one per candidate
    scoring : list of _ResolvedMetric, the metrics evaluated
    multimetric : bool
        whether multiple metrics were requested. If False, score keys are suffixed
        with ``"score"``, if True, with the name of the respective metric.

    Returns
    -------
    dict, with str keys, the ``cv_results_`` attribute of the tuner
    """
    params = [candidate["params"] for candidate in candidates]

    results = {"params": params}
    results.update(_param_columns(params))

    for key in ["fit_time", "score_time"]:
        times = np.array([candidate[key] for candidate in candidates], dtype=float)
        results[f"mean_{key}"] = np.mean(times, axis=1)
        results[f"std_{key}"] = np.std(times, axis=1)

    for entry in scoring:
        suffix = entry.name if multimetric else "score"
        scores = np.array(
            [candidate["scores"][entry.name] for candidate in candidates], dtype=float
        )

        for i in range(scores.shape[1]):
            results[f"split{i}_test_{suffix}"] = scores[:, i]

        mean_scores = np.mean(scores, axis=1)
        results[f"mean_test_{suffix}"] = mean_scores
        results[f"std_test_{suffix}"] = np.std(scores, axis=1)
        results[f"rank_test_{suffix}"] = _rank_scores(
            mean_scores, entry.greater_is_better
        )

    return results


def _check_refit(refit, scoring, multimetric):
    """Check that refit selects one of the metrics, raise ValueError if not."""
    if not multimetric or not isinstance(refit, str):
        return

    names = [entry.name for entry in scoring]
    if refit not in names:
        raise ValueError(
            "Error in grid search tuner, for multi-metric scoring, a string refit "
            f"must be one of the metric names {names}, but found {refit!r}"
        )


def _select_best_index(cv_results, scoring, multimetric, refit, estimator):
    """Select the best candidate, and the cv_results_ key of its score.

    Parameters
    ----------
    cv_results : dict, as returned by ``_make_cv_results``
    scoring : list of _ResolvedMetric, the metrics evaluated
    multimetric : bool, whether multiple metrics were requested
    refit : bool, str, or callable
        if callable, is applied to ``cv_results`` to obtain the best index.
        If a str and ``multimetric``, names the metric to select by.
        Otherwise, the first metric in ``scoring`` is used to select.
    estimator : the estimator tuned, used in the error message only

    Returns
    -------
    best_index : int, index of the best candidate in ``cv_results``
    score_key : str or None, key of the best score, None if ``refit`` is callable
    """
    if callable(refit):
        return int(refit(cv_results)), None

    if multimetric:
        suffix = refit if isinstance(refit, str) else scoring[0].name
    else:
        suffix = "score"

    score_key = f"mean_test_{suffix}"
    if np.isnan(cv_results[score_key]).all():
        raise NotFittedError(
            "Error in grid search tuner, all fits of the estimator failed, "
            "set error_score='raise' to see the exceptions. "
            f"Failed estimator: {estimator}"
        )

    return int(np.argmin(cv_results[f"rank_test_{suffix}"])), score_key


def _run_grid_search(
    estimator,
    param_grid,
    X,
    y,
    estimator_type,
    cv=None,
    scoring=None,
    greater_is_better="auto",
    refit=True,
    error_score=np.nan,
    backend=None,
    backend_params=None,
    verbose=0,
):
    """Run grid search over param_grid, by backtesting via native evaluate.

    Each parameter candidate in ``param_grid`` is backtested via
    ``model_evaluation.evaluate``, on identical folds, and the candidate with the
    best mean test score is selected. Candidates are evaluated in parallel,
    as per ``backend`` and ``backend_params``.

    Parameters
    ----------
    estimator : sktime classifier or regressor, the estimator to tune
    param_grid : dict or list of dict, the parameter grid to search over
    X : sktime compatible panel data, the training features
    y : 1D or 2D np.ndarray, the training targets
    estimator_type : str, one of "classifier" or "regressor"
    cv : int, cross-validation generator, iterable of splits, or None
    scoring : None, str, callable, sklearn scorer, or list or dict of these
    greater_is_better : "auto", bool, optional, default="auto"
    refit : bool, str, or callable, optional, default=True
    error_score : "raise" or numeric, optional, default=np.nan
    backend : str, optional, parallelization backend, see ``utils.parallel``
    backend_params : dict, optional, parameters passed to the backend
    verbose : int, optional, default=0, if positive, prints the number of fits

    Returns
    -------
    dict, the fitted attributes to write to the tuner, with keys
    ``cv_results_``, ``best_index_``, ``best_params_``, ``n_splits_``,
    ``scorer_``, ``multimetric_``, and ``best_score_`` unless ``refit``
    is a callable
    """
    multimetric = isinstance(scoring, (list, tuple, dict))
    resolved = _resolve_scoring(scoring, estimator_type, greater_is_better)
    _check_refit(refit, resolved, multimetric)

    _check_param_grid(param_grid)
    candidate_params = list(ParameterGrid(param_grid))
    if len(candidate_params) == 0:
        raise ValueError(
            "Error in grid search tuner, no parameter candidates to evaluate, "
            "param_grid is empty."
        )

    y = _coerce_y(y)
    cv = _resolve_cv(cv, y, estimator_type)
    n_splits = cv.get_n_splits()

    if verbose > 0:
        print(
            f"Fitting {n_splits} folds for each of {len(candidate_params)} candidates,"
            f" totalling {len(candidate_params) * n_splits} fits"
        )

    meta = {
        "estimator": estimator,
        "X": X,
        "y": y,
        "cv": cv,
        "scoring": resolved,
        "error_score": error_score,
    }

    candidates = parallelize(
        fun=_fit_and_score,
        iter=candidate_params,
        meta=meta,
        backend=backend,
        backend_params=backend_params,
    )

    cv_results = _make_cv_results(candidates, resolved, multimetric)
    best_index, score_key = _select_best_index(
        cv_results, resolved, multimetric, refit, estimator
    )

    if multimetric:
        scorer = {entry.name: entry.metric for entry in resolved}
    else:
        scorer = resolved[0].metric

    fitted_params = {
        "cv_results_": cv_results,
        "best_index_": best_index,
        "best_params_": cv_results["params"][best_index],
        "n_splits_": n_splits,
        "scorer_": scorer,
        "multimetric_": multimetric,
    }
    if score_key is not None:
        fitted_params["best_score_"] = cv_results[score_key][best_index]

    return fitted_params


def _fit_and_time(estimator, X, y):
    """Fit estimator to the full data, and return the seconds taken."""
    start = time.perf_counter()
    estimator.fit(X=X, y=_coerce_y(y))
    return time.perf_counter() - start


# backends of parallelize that take joblib.Parallel keyword arguments
_JOBLIB_BACKENDS = ["loky", "multiprocessing", "threading", "joblib"]


# todo 1.3.0: remove this function, and use tuner.backend and tuner.backend_params
# directly in _fit_tuner, together with removal of the n_jobs and pre_dispatch
# parameters of TSCGridSearchCV and TSRGridSearchCV
def _resolve_deprecated_parallel(tuner):
    """Resolve the deprecated n_jobs and pre_dispatch parameters of a tuner.

    Values passed are forwarded to the joblib backend via ``backend_params``,
    overriding values present there, so that they keep working as before.
    If no ``backend`` is set, the ``loky`` backend is selected, as the
    parameters have no effect on the sequential default.

    Parameters
    ----------
    tuner : TSCGridSearchCV or TSRGridSearchCV instance

    Returns
    -------
    backend : str or None, backend to pass to ``parallelize``
    backend_params : dict or None, backend parameters to pass to ``parallelize``
    """
    backend = tuner.backend
    backend_params = tuner.backend_params

    deprecated = {
        name: getattr(tuner, name)
        for name in ["n_jobs", "pre_dispatch"]
        if getattr(tuner, name) != "deprecated"
    }
    if len(deprecated) == 0:
        return backend, backend_params

    cls_name = type(tuner).__name__
    passed = ", ".join(f"{name}={value!r}" for name, value in deprecated.items())

    if backend is not None and backend not in _JOBLIB_BACKENDS:
        warn(
            f"Parameters n_jobs and pre_dispatch of {cls_name} are deprecated "
            "and will be removed in sktime 1.3.0. The values passed "
            f"({passed}) apply to joblib backends only, and are ignored for "
            f"backend={backend!r}. Pass parallelization parameters in "
            "backend_params instead.",
            DeprecationWarning,
            obj=tuner,
        )
        return backend, backend_params

    warn(
        f"Parameters n_jobs and pre_dispatch of {cls_name} are deprecated and "
        "will be removed in sktime 1.3.0. The values passed "
        f"({passed}) are forwarded to the joblib backend for now. To retain "
        "current behaviour and silence this warning, pass them via backend and "
        f"backend_params instead, e.g., backend='loky', "
        f"backend_params={deprecated!r}.",
        DeprecationWarning,
        obj=tuner,
    )

    backend_params = dict(backend_params) if backend_params else {}
    backend_params.update(deprecated)
    if backend is None:
        backend = "loky"

    return backend, backend_params


def _fit_tuner(tuner, X, y, estimator_type):
    """Run the grid search for a tuner, and write the results to it.

    Shared ``_fit`` logic of ``TSCGridSearchCV`` and ``TSRGridSearchCV``.

    Parameters
    ----------
    tuner : TSCGridSearchCV or TSRGridSearchCV instance
    X : sktime compatible panel data, the training features
    y : 1D or 2D np.ndarray, the training targets
    estimator_type : str, one of "classifier" or "regressor"

    Returns
    -------
    tuner : reference to ``tuner``, with the fitted attributes written
    """
    backend, backend_params = _resolve_deprecated_parallel(tuner)

    results = _run_grid_search(
        estimator=tuner.estimator,
        param_grid=tuner.param_grid,
        X=X,
        y=y,
        estimator_type=estimator_type,
        cv=tuner.cv,
        scoring=tuner.scoring,
        greater_is_better=tuner.greater_is_better,
        refit=tuner.refit,
        error_score=tuner.error_score,
        backend=backend,
        backend_params=backend_params,
        verbose=tuner.verbose,
    )
    for name, value in results.items():
        setattr(tuner, name, value)

    tuner.best_estimator_ = tuner.estimator.clone().set_params(**tuner.best_params_)
    if tuner.refit:
        tuner.refit_time_ = _fit_and_time(tuner.best_estimator_, X, y)

        # sklearn compatible attributes, set only if the tuned estimator has them
        for attr in ["n_features_in_", "feature_names_in_"]:
            if hasattr(tuner.best_estimator_, attr):
                setattr(tuner, attr, getattr(tuner.best_estimator_, attr))

    return tuner


def _check_refit_for_predict(tuner):
    """Raise a RuntimeError if the tuner cannot predict, because refit is False."""
    if not tuner.refit:
        name = type(tuner).__name__
        raise RuntimeError(
            f"In {name}, refit must be True to make predictions, but found "
            f"refit=False. If refit=False, {name} can be used only to tune "
            "hyper-parameters, as a parameter estimator."
        )


def _coerce_prediction(y_pred):
    """Coerce a prediction of the tuned estimator to 2D np.ndarray."""
    if hasattr(y_pred, "to_numpy"):
        y_pred = y_pred.to_numpy()
    y_pred = np.asarray(y_pred)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    return y_pred


# todo 1.3.0: remove the n_jobs and pre_dispatch parameters, from the
# signature and the docstring, and remove the call to
# _resolve_deprecated_parallel in _fit_tuner
class TSCGridSearchCV(_DelegatedClassifier):
    """Exhaustive search over specified parameter values for a classifier.

    Optimizes hyper-parameters of ``estimator`` by exhaustive grid search, using
    ``sktime`` native backtesting via
    ``classification.model_evaluation.evaluate``.

    In ``fit``, each parameter combination in ``param_grid`` is backtested on the
    data passed, using the cross-validation scheme ``cv`` and the metric
    ``scoring``. All candidates are evaluated on identical folds.

    The parameter combination with the best mean test score is set as
    ``best_params_``, and a clone of ``estimator`` with those parameters is set as
    ``best_estimator_``. If ``refit`` is not False, ``best_estimator_`` is fitted
    to the entire data, and ``predict`` and ``predict``-like methods of the tuner
    call the respective method of ``best_estimator_``.

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

    scoring : None, str, callable, sklearn scorer, or list or dict of these
        Metric or metrics to evaluate the cross-validated model with.

        - a callable must have signature ``(y_true, y_pred) -> float``, e.g.,
          ``accuracy_score`` from ``sklearn.metrics``. Its value is reported as is
        - a string must name a scikit-learn scorer, e.g., ``"accuracy"``. Values
          are reported with the sign convention of the scorer, so values of
          ``"neg_log_loss"`` are negative
        - a list or dict selects multiple metrics. The first is used to rank
          candidates, unless ``refit`` names another. Dict keys are used as the
          metric names in ``cv_results_``
        - if None, defaults to ``accuracy_score``

    n_jobs : int, optional, default="deprecated"
        Number of jobs to run in parallel over the parameter candidates.

        Deprecated, and will be removed in sktime 1.3.0. If passed, the value is
        written to ``backend_params``, and ``backend`` defaults to ``"loky"``,
        so behaviour is unchanged. To retain the behaviour after removal, pass
        ``backend="loky"`` and ``backend_params={"n_jobs": ...}`` instead.

    refit : bool, str, or callable, default=True
        Refit ``best_estimator_`` using the best found parameters on the whole
        dataset. If False, ``predict`` and ``predict``-like methods raise, and
        the tuner can be used only to tune hyper-parameters, e.g., as a
        parameter estimator via ``get_fitted_params``.

        For multi-metric evaluation, this can be a ``str`` naming the metric to
        select the best parameters by.

        Where there are considerations other than the best score in choosing the
        best parameters, ``refit`` can be a callable, which is applied to
        ``cv_results_`` and returns the selected ``best_index_``. In that case
        ``best_score_`` is not available.

        The refitted estimator is available at the ``best_estimator_``
        attribute, and permits calling ``predict`` directly on the tuner.

    cv : int, cross-validation generator, iterable of splits, or None, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds in a ``(Stratified)KFold``,
        - a cross-validation splitter with a ``split`` method,
        - an iterable yielding (train, test) splits as arrays of indices.

        For integer and None inputs, ``StratifiedKFold`` is used if ``y`` is
        binary or multiclass, and ``KFold`` otherwise. Both are instantiated with
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
        and are available in fields of the classifiers_ attribute.
        Has the same effect as applying ColumnEnsembleClassifier wrapper to self.
        If False, the same best parameter is selected for all variables.

    greater_is_better : "auto", bool, optional, default="auto"
        Whether higher values of the reported metric are better, used to rank
        the parameter candidates.

        - "auto" determines the direction from the metric. Scikit-learn scorers
          are higher-is-better by their sign convention. For metric callables,
          the direction is inferred from the metric, e.g., ``accuracy_score`` is
          higher-is-better, and ``log_loss`` is lower-is-better
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
        Clone of ``estimator`` with the best found parameters set, i.e., the
        parameters which gave the best mean test score on the held out data.
        Fitted to the entire data if ``refit`` is not False, otherwise unfitted.
        See the ``refit`` parameter for more information on allowed values.

    best_score_ : float
        Mean cross-validated score of ``best_estimator_``.
        Not available if ``refit`` is a callable.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

    best_index_ : int
        The index in the ``cv_results_`` arrays which corresponds to the best
        candidate parameter setting.

        The dict at ``cv_results_["params"][best_index_]`` gives the parameter
        setting for the best model, i.e., is identical with ``best_params_``.

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

    n_features_in_ : int
        Number of features seen during ``fit``.
        Present only if ``refit`` is not False, and ``best_estimator_``
        exposes ``n_features_in_`` after being fitted.

    feature_names_in_ : ndarray of shape (``n_features_in_``,)
        Names of features seen during ``fit``.
        Present only if ``refit`` is not False, and ``best_estimator_``
        exposes ``feature_names_in_`` after being fitted.

    classes_ : ndarray of shape (n_classes,)
        The class labels seen in ``fit``.

    See Also
    --------
    ParameterGrid : Generates all the combinations of a hyperparameter grid.
    sktime.classification.model_evaluation.evaluate : Backtesting used internally.
    sklearn.metrics.make_scorer : Make a scorer from a performance metric or
        loss function.

    Examples
    --------
    >>> from sklearn.metrics import accuracy_score
    >>> from sktime.classification.dummy import DummyClassifier
    >>> from sktime.classification.model_selection import TSCGridSearchCV
    >>> from sktime.datasets import load_unit_test
    >>>
    >>> X, y = load_unit_test(split="train")
    >>> tuned = TSCGridSearchCV(
    ...     DummyClassifier(),
    ...     param_grid={"strategy": ["most_frequent", "prior"]},
    ...     scoring=accuracy_score,
    ...     cv=2,
    ... )
    >>> tuned = tuned.fit(X, y)
    >>> y_pred = tuned.predict(X)
    >>> best_params = tuned.best_params_
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["fkiraly", "achieveordie", "yash-sangwan"],
        # estimator type
        # --------------
        "X_inner_mtype": "pd-multiindex",
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
    #     all non-overridden methods are same as of getattr(self, _delegate_name)
    #     see further details in _DelegatedClassifier docstring
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
        """Fit time series classifier to training data.

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
            2D np.ndarray of shape [n_instances, n_outputs], class labels

        Returns
        -------
        self : Reference to self.
        """
        return _fit_tuner(self, X, y, estimator_type="classifier")

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
        y : 2D np.ndarray of shape [n_instances, n_outputs], predicted class labels
        """
        _check_refit_for_predict(self)
        return _coerce_prediction(self._get_delegate().predict(X=X))

    def _predict_proba(self, X):
        """Predict class probabilities for sequences in X.

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
        _check_refit_for_predict(self)
        return super()._predict_proba(X)

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
