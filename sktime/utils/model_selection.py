# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Internal helpers for native model-selection estimators."""

from inspect import signature
from numbers import Integral
from time import perf_counter

import numpy as np
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, ParameterGrid

from sktime.utils.parallel import parallelize


class _EvaluateCV:
    """Adapt scikit-learn CV inputs to ``evaluate``'s one-argument splitter."""

    def __init__(self, cv, y):
        self.cv = cv
        self.y = np.asarray(y)
        if self.y.ndim > 1 and self.y.shape[1] == 1:
            self.y = self.y.ravel()

        if hasattr(cv, "split"):
            self._splits = None
        else:
            self._splits = list(cv)

    def split(self, X):
        if self._splits is not None:
            return iter(self._splits)

        try:
            return self.cv.split(X)
        except TypeError:
            return self.cv.split(X, self.y)

    def get_n_splits(self):
        if self._splits is not None:
            return len(self._splits)
        try:
            return self.cv.get_n_splits()
        except TypeError:
            return self.cv.get_n_splits(None, self.y)


class _TrainCV:
    """Turn a CV splitter into one that scores on each training fold."""

    def __init__(self, cv):
        self.cv = cv

    def split(self, X):
        for train, _ in self.cv.split(X):
            yield train, train

    def get_n_splits(self):
        return self.cv.get_n_splits()


class _NamedMetric:
    """Pickleable wrapper for a metric with a caller-selected name."""

    def __init__(self, metric, name):
        self.name = name

        if isinstance(metric, str):
            metric = get_scorer(metric)

        if hasattr(metric, "_score_func"):
            self.metric = metric._score_func
            self.kwargs = metric._kwargs
            self.sign = metric._sign
        else:
            self.metric = metric
            self.kwargs = {}
            self.sign = 1

        metric_name = getattr(self.metric, "__name__", name)
        self.__name__ = metric_name
        self.__qualname__ = metric_name
        try:
            self.__signature__ = signature(self.metric)
        except (TypeError, ValueError):
            pass

    def __call__(self, y_true, y_pred, **kwargs):
        if (
            self.__name__ == "roc_auc_score"
            and getattr(y_pred, "ndim", 1) == 2
            and y_pred.shape[1] == 2
            and len(np.unique(y_true)) == 2
        ):
            y_pred = y_pred[:, 1]
        score_kwargs = self.kwargs.copy()
        score_kwargs.update(kwargs)
        return self.sign * self.metric(y_true, y_pred, **score_kwargs)


def _make_evaluate_cv(cv, y):
    """Normalize a CV parameter for repeated calls to ``evaluate``."""
    if cv is None:
        cv = KFold(n_splits=3, shuffle=False)
    elif isinstance(cv, Integral):
        cv = KFold(n_splits=cv, shuffle=False)
    return _EvaluateCV(cv, y)


def _named_metric(metric, name):
    """Return a callable with the stable name expected by ``evaluate``."""
    if not isinstance(metric, str) and not callable(metric):
        raise TypeError(
            "scoring must be None, a callable, a scikit-learn scorer, "
            "a string, a list/tuple, or a dictionary"
        )

    if (
        not hasattr(metric, "_score_func")
        and getattr(metric, "__name__", None) == name
        and name not in {"brier_score_loss", "log_loss", "roc_auc_score"}
    ):
        return metric

    return _NamedMetric(metric, name)


def _resolve_scorers(scoring, default):
    """Resolve legacy scoring forms to callables accepted by ``evaluate``."""
    if scoring is None:
        scoring = default

    if isinstance(scoring, dict):
        items = list(scoring.items())
    elif isinstance(scoring, (list, tuple)):
        items = [
            (_get_metric_name(metric, f"score_{i}"), metric)
            for i, metric in enumerate(scoring)
        ]
    else:
        items = [(_get_metric_name(scoring, str(scoring)), scoring)]

    names = [str(name) for name, _ in items]
    if len(names) != len(set(names)):
        raise ValueError("scoring metrics must have unique names")

    scorers = {
        name: _named_metric(metric, name) for name, (_, metric) in zip(names, items)
    }
    metric_names = [metric.__name__ for metric in scorers.values()]
    if len(metric_names) != len(set(metric_names)):
        raise ValueError("scoring metrics must have unique callable names")
    return scorers


def _get_metric_name(metric, default):
    """Get a stable name for a callable or scikit-learn scorer."""
    if isinstance(metric, str):
        return metric
    if hasattr(metric, "__name__"):
        return metric.__name__
    if hasattr(metric, "_score_func"):
        return getattr(metric._score_func, "__name__", default)
    return default


def _fit_and_score(params, meta):
    """Fit one parameter candidate and evaluate it with native ``evaluate``."""
    estimator = meta["estimator"].clone().set_params(**params)
    evaluate_kwargs = {
        "cv": meta["cv"],
        "X": meta["X"],
        "y": meta["y"],
        "scoring": list(meta["scorers"].values()),
        "return_data": False,
        "error_score": meta["error_score"],
    }
    test_results = meta["evaluate"](estimator, **evaluate_kwargs)

    result = {
        "params": params,
        "test": {
            name: np.asarray(test_results[f"test_{metric.__name__}"], dtype=float)
            for name, metric in meta["scorers"].items()
        },
        "fit_time": np.asarray(test_results["fit_time"], dtype=float),
        "score_time": _get_score_time(test_results),
    }

    if meta["return_train_score"]:
        train_results = meta["evaluate"](
            estimator,
            **{
                **evaluate_kwargs,
                "cv": _TrainCV(meta["cv"]),
            },
        )
        result["train"] = {
            name: np.asarray(train_results[f"test_{metric.__name__}"], dtype=float)
            for name, metric in meta["scorers"].items()
        }

    return result


def _get_score_time(results):
    """Aggregate prediction times for deterministic and probabilistic metrics."""
    time_columns = [
        column
        for column in results.columns
        if column.endswith("_time") and column != "fit_time"
    ]
    if not time_columns:
        return np.zeros(len(results), dtype=float)
    return results[time_columns].sum(axis=1).to_numpy(dtype=float)


def _run_grid_search(
    estimator,
    param_grid,
    cv,
    scorers,
    evaluate,
    X,
    y,
    error_score,
    n_jobs,
    pre_dispatch,
    verbose,
    return_train_score,
):
    """Run candidates through native ``evaluate`` and return raw results."""
    candidates = list(ParameterGrid(param_grid))
    if len(candidates) == 0:
        raise ValueError("No fits were performed. The parameter grid is empty.")

    n_splits = cv.get_n_splits()
    if verbose > 0:
        print(
            f"Fitting {n_splits} folds for each of {len(candidates)} candidates, "
            f"totalling {len(candidates) * n_splits} fits"
        )

    backend = None
    backend_params = {}
    if n_jobs not in (None, 1):
        backend = "loky"
        backend_params["n_jobs"] = n_jobs
        if pre_dispatch is not None:
            backend_params["pre_dispatch"] = pre_dispatch

    meta = {
        "estimator": estimator,
        "cv": cv,
        "scorers": scorers,
        "evaluate": evaluate,
        "X": X,
        "y": y,
        "error_score": error_score,
        "return_train_score": return_train_score,
    }
    return parallelize(
        fun=_fit_and_score,
        iter=candidates,
        meta=meta,
        backend=backend,
        backend_params=backend_params,
    )


def _rank_scores(scores):
    """Rank scores in descending order, putting NaNs last."""
    scores = np.asarray(scores, dtype=float)
    ranks = np.full(scores.shape, scores.size + 1, dtype=int)
    valid = ~np.isnan(scores)
    if not valid.any():
        return ranks

    valid_indices = np.flatnonzero(valid)
    order = valid_indices[np.argsort(-scores[valid], kind="mergesort")]
    previous = None
    rank = 0
    for position, index in enumerate(order, start=1):
        if previous is None or scores[index] != previous:
            rank = position
            previous = scores[index]
        ranks[index] = rank
    return ranks


def _make_cv_results(raw_results, scorers, return_train_score):
    """Convert native evaluation output to a GridSearchCV-like result dict."""
    metric_names = list(scorers)
    n_splits = len(raw_results[0]["test"][metric_names[0]])

    results = {
        "params": np.asarray(
            [result["params"] for result in raw_results], dtype=object
        ),
        "mean_fit_time": np.asarray(
            [np.mean(result["fit_time"]) for result in raw_results]
        ),
        "std_fit_time": np.asarray(
            [np.std(result["fit_time"]) for result in raw_results]
        ),
        "mean_score_time": np.asarray(
            [np.mean(result["score_time"]) for result in raw_results]
        ),
        "std_score_time": np.asarray(
            [np.std(result["score_time"]) for result in raw_results]
        ),
    }

    param_names = sorted({name for result in raw_results for name in result["params"]})
    for param_name in param_names:
        results[f"param_{param_name}"] = np.asarray(
            [result["params"].get(param_name, np.nan) for result in raw_results],
            dtype=object,
        )

    for metric_name in metric_names:
        test_scores = np.asarray(
            [result["test"][metric_name] for result in raw_results], dtype=float
        )
        for split in range(n_splits):
            results[f"split{split}_test_{metric_name}"] = test_scores[:, split]

        mean_key = f"mean_test_{metric_name}"
        results[mean_key] = np.mean(test_scores, axis=1)
        results[f"std_test_{metric_name}"] = np.std(test_scores, axis=1)
        results[f"rank_test_{metric_name}"] = _rank_scores(results[mean_key])

        if return_train_score:
            train_scores = np.asarray(
                [result["train"][metric_name] for result in raw_results], dtype=float
            )
            for split in range(n_splits):
                results[f"split{split}_train_{metric_name}"] = train_scores[:, split]
            results[f"mean_train_{metric_name}"] = np.mean(train_scores, axis=1)
            results[f"std_train_{metric_name}"] = np.std(train_scores, axis=1)

    if len(metric_names) == 1:
        metric_name = metric_names[0]
        aliases = {
            key.replace(f"_{metric_name}", "_score"): value
            for key, value in results.items()
            if key.startswith(("split", "mean", "std", "rank"))
            and f"_{metric_name}" in key
        }
        results.update(aliases)

    return results


def _select_best_index(cv_results, scoring_names, refit):
    """Select the best candidate and its score column."""
    if callable(refit) and not isinstance(refit, str):
        return int(refit(cv_results)), None

    if isinstance(refit, str):
        if refit not in scoring_names:
            raise ValueError(
                f"refit={refit!r} is not one of the scoring metrics: {scoring_names}"
            )
        metric_name = refit
    else:
        if len(scoring_names) > 1 and refit:
            raise ValueError(
                "For multi-metric scoring, refit must be a metric name, "
                "False, or a callable"
            )
        metric_name = scoring_names[0]

    score_key = f"mean_test_{metric_name}"
    scores = np.asarray(cv_results[score_key], dtype=float)
    if np.isnan(scores).all():
        raise ValueError(
            "All fits failed. Set error_score='raise' to see the underlying error."
        )
    return int(np.nanargmax(scores)), score_key


def _refit_estimator(estimator, params, X, y):
    """Clone and fit the selected estimator, returning it and elapsed time."""
    best_estimator = estimator.clone().set_params(**params)
    start = perf_counter()
    best_estimator.fit(X=X, y=y)
    return best_estimator, perf_counter() - start
