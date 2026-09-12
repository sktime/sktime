"""Grid search engine shared by the classification and regression tuners.

Contains the parameter candidate loop, the pinning of cross-validation folds,
and the assembly of sklearn style ``cv_results_``. Used by ``TSCGridSearchCV``
and ``TSRGridSearchCV``.
"""

__author__ = ["fkiraly", "achieveordie", "yash-sangwan"]

import time

import numpy as np
from sklearn.model_selection import ParameterGrid

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
    # deferred import, base must not import a type module at module level,
    # see sktime/tests/test_cross_module_imports.py
    from sktime.classification.model_evaluation import evaluate

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


# todo 1.3.0: remove this function and its call in _fit_tuner, together with
# removal of the return_train_score parameter of TSCGridSearchCV and
# TSRGridSearchCV
def _check_return_train_score(tuner):
    """Warn if the deprecated return_train_score parameter of a tuner is set."""
    if not tuner.return_train_score:
        return

    warn(
        f"Parameter return_train_score of {type(tuner).__name__} is deprecated "
        "and will be removed in sktime 1.3.0. Train scores are not computed by "
        "the native grid search, so the value passed is ignored, and "
        "cv_results_ contains test scores only.",
        DeprecationWarning,
        obj=tuner,
    )


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
    _check_return_train_score(tuner)

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
