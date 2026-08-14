"""Resolution of sklearn scoring specifications for sktime tuners.

Turns the ``scoring`` argument of the sktime tuners into metric callables that
``model_evaluation.evaluate`` can consume, together with the sign and ranking
direction needed to select the best parameter candidate.
"""
# copyright/attribution hyperactive developers, MIT License
# _guess_greater_is_better is a port of _guess_sign_of_sklmetric from
# hyperactive.experiment.integrations._skl_metrics, returning a bool instead of a sign

__author__ = ["fkiraly", "yash-sangwan"]
__all__ = ["_ResolvedMetric", "_guess_greater_is_better", "_resolve_scoring"]

from collections import namedtuple
from inspect import signature

_HIGHER_IS_BETTER = {
    # classification
    "accuracy_score": True,
    "auc": True,
    "average_precision_score": True,
    "balanced_accuracy_score": True,
    "brier_score_loss": False,
    "class_likelihood_ratios": False,
    "cohen_kappa_score": True,
    "d2_log_loss_score": True,
    "dcg_score": True,
    "f1_score": True,
    "fbeta_score": True,
    "hamming_loss": False,
    "hinge_loss": False,
    "jaccard_score": True,
    "log_loss": False,
    "matthews_corrcoef": True,
    "ndcg_score": True,
    "precision_score": True,
    "recall_score": True,
    "roc_auc_score": True,
    "top_k_accuracy_score": True,
    "zero_one_loss": False,
    # regression
    "d2_absolute_error_score": True,
    "d2_pinball_score": True,
    "d2_tweedie_score": True,
    "explained_variance_score": True,
    "max_error": False,
    "mean_absolute_error": False,
    "mean_absolute_percentage_error": False,
    "mean_gamma_deviance": False,
    "mean_pinball_loss": False,
    "mean_poisson_deviance": False,
    "mean_squared_error": False,
    "mean_squared_log_error": False,
    "mean_tweedie_deviance": False,
    "median_absolute_error": False,
    "r2_score": True,
    "root_mean_squared_error": False,
    "root_mean_squared_log_error": False,
}


_ResolvedMetric = namedtuple(
    "_ResolvedMetric", ["name", "metric", "sign", "greater_is_better"]
)
_ResolvedMetric.__doc__ = """One resolved entry of a scoring specification.

name : str
    name of the metric, used in the ``cv_results_`` keys of the tuners
metric : callable
    metric with signature ``(y_true, y_pred) -> float``, passed to ``evaluate``
sign : int
    1 or -1, multiplier applied to the value returned by ``metric``,
    to obtain the value reported in ``cv_results_``
greater_is_better : bool
    whether higher values of ``sign * metric`` indicate better performance
"""


def _guess_greater_is_better(metric):
    """Guess whether higher values of a sklearn metric are better.

    Parameters
    ----------
    metric : callable
        sklearn metric function to guess the direction for.

    Returns
    -------
    bool
        True if higher values are better, False if lower values are better.
    """
    metric_name = getattr(metric, "__name__", None)

    if hasattr(metric, "greater_is_better"):
        return bool(metric.greater_is_better)
    if metric_name is None:
        # no name available, conservatively assume lower is better
        return False
    if metric_name in _HIGHER_IS_BETTER:
        return _HIGHER_IS_BETTER[metric_name]
    if metric_name.endswith("_score"):
        return True
    if metric_name.endswith(("_loss", "_deviance", "_error")):
        return False
    # if the direction cannot be determined, assume lower is better
    return False


def _default_metric(estimator_type):
    """Get the default metric for an estimator type.

    Parameters
    ----------
    estimator_type : str, one of "classifier" or "regressor"

    Returns
    -------
    callable, the default metric for ``estimator_type``
    """
    from sklearn.metrics import accuracy_score, r2_score

    if estimator_type == "classifier":
        return accuracy_score
    if estimator_type == "regressor":
        return r2_score
    raise ValueError(
        'estimator_type must be one of "classifier" or "regressor", '
        f"but found {estimator_type!r}"
    )


class _MetricAdapter:
    """Metric unwrapped from a sklearn scorer, with its scorer kwargs bound.

    Presents the name and signature of the wrapped metric, so that ``evaluate``
    identifies it in the same way as the metric itself.

    Parameters
    ----------
    metric : callable
        metric with signature ``(y_true, y_pred, **kwargs) -> float``
    kwargs : dict
        keyword arguments to bind, the ``_kwargs`` of the scorer
    """

    def __init__(self, metric, kwargs):
        self.metric = metric
        self.kwargs = kwargs
        self.__name__ = metric.__name__
        try:
            self.__signature__ = signature(metric)
        except (TypeError, ValueError):
            # no introspectable signature, callers then see that of __call__
            pass

    def __call__(self, y_true, y_pred, **kwargs):
        return self.metric(y_true, y_pred, **{**self.kwargs, **kwargs})


def _unwrap_scorer(scorer):
    """Unwrap a sklearn scorer into a metric callable and its sign.

    Parameters
    ----------
    scorer : sklearn scorer, as returned by ``get_scorer`` or ``make_scorer``

    Returns
    -------
    metric : callable, the metric wrapped by ``scorer``
    sign : int, 1 or -1, the sign convention of ``scorer``
    """
    score_func = getattr(scorer, "_score_func", None)
    if score_func is None:
        raise TypeError(
            "Error in sktime tuner, scoring could not be resolved to a metric "
            f"with signature (y_true, y_pred) -> float, found scorer {scorer!r} "
            "without a wrapped metric. Pass the metric callable instead."
        )
    metric = _MetricAdapter(score_func, getattr(scorer, "_kwargs", {}))
    return metric, getattr(scorer, "_sign", 1)


def _takes_estimator(scoring):
    """Check whether a callable takes an estimator argument, i.e., is scorer style."""
    try:
        return "estimator" in signature(scoring).parameters
    except (TypeError, ValueError):
        return False


def _resolve_one(scoring, estimator_type):
    """Resolve a single scoring specification.

    Parameters
    ----------
    scoring : None, str, callable, or sklearn scorer
    estimator_type : str, one of "classifier" or "regressor"

    Returns
    -------
    name : str, name of the metric
    metric : callable, metric with signature ``(y_true, y_pred) -> float``
    sign : int, 1 or -1, multiplier to apply to the value of ``metric``
    from_scorer : bool, whether ``scoring`` was a sklearn scorer or scorer name
    """
    from sklearn.metrics import get_scorer

    if scoring is None:
        metric = _default_metric(estimator_type)
        return metric.__name__, metric, 1, False

    if isinstance(scoring, str):
        metric, sign = _unwrap_scorer(get_scorer(scoring))
        return scoring, metric, sign, True

    if hasattr(scoring, "_score_func"):
        metric, sign = _unwrap_scorer(scoring)
        return metric.__name__, metric, sign, True

    if not callable(scoring):
        raise TypeError(
            "Error in sktime tuner, scoring must be None, a string, a metric "
            "callable, a sklearn scorer, or a list or dict of these, "
            f"but found {scoring!r}"
        )

    if _takes_estimator(scoring):
        raise TypeError(
            "Error in sktime tuner, scoring callables must have signature "
            "(y_true, y_pred) -> float, e.g., accuracy_score from sklearn.metrics, "
            f"but {scoring!r} takes an estimator argument. Scorer style callables "
            "are not supported, pass the metric or its scorer name instead."
        )

    return getattr(scoring, "__name__", "score"), scoring, 1, False


def _resolve_scoring(scoring, estimator_type, greater_is_better="auto"):
    """Resolve a scoring specification to metrics and ranking directions.

    Values reported by the tuners are ``sign * metric(y_true, y_pred)``.
    For metric callables the sign is 1, so the metric value is reported as is.
    For sklearn scorers and scorer names the sign is that of the scorer, so that
    the reported value follows the sklearn convention, e.g., the value reported
    for ``"neg_mean_squared_error"`` is negative.

    Parameters
    ----------
    scoring : None, str, callable, sklearn scorer, or list or dict of these
        metrics to evaluate. If a list or dict, the first entry is the primary
        metric, used to rank candidates. Dict keys are used as metric names.
        If None, defaults to ``accuracy_score`` for classifiers,
        and ``r2_score`` for regressors.

    estimator_type : str, one of "classifier" or "regressor"
        type of the estimator to tune, determines the default metric.

    greater_is_better : "auto", bool, optional, default="auto"
        whether higher values of the reported metric are better.
        If "auto", is determined from the metric, via ``_guess_greater_is_better``
        for metric callables, and from the sign convention for sklearn scorers.

    Returns
    -------
    list of _ResolvedMetric
        one entry per metric, the first entry is the primary metric.

    Raises
    ------
    TypeError
        if ``scoring`` or one of its entries cannot be resolved to a metric
    ValueError
        if ``scoring`` is empty, if metric names are not unique,
        or if ``greater_is_better`` is not "auto" or a bool
    """
    if greater_is_better != "auto" and not isinstance(greater_is_better, bool):
        raise ValueError(
            'Error in sktime tuner, greater_is_better must be "auto", True, or False, '
            f"but found {greater_is_better!r}"
        )

    if isinstance(scoring, dict):
        items = list(scoring.items())
    elif isinstance(scoring, (list, tuple)):
        items = [(None, entry) for entry in scoring]
    else:
        items = [(None, scoring)]

    if len(items) == 0:
        raise ValueError("Error in sktime tuner, scoring must not be empty")

    resolved = []
    for key, entry in items:
        name, metric, sign, from_scorer = _resolve_one(entry, estimator_type)
        if greater_is_better == "auto":
            # scorers are higher-is-better by construction, after their sign
            direction = True if from_scorer else _guess_greater_is_better(metric)
        else:
            direction = greater_is_better
        resolved.append(_ResolvedMetric(key or name, metric, sign, direction))

    _check_unique_names(resolved)

    return resolved


def _check_unique_names(resolved):
    """Check that resolved metrics can be told apart, raise ValueError if not."""
    names = [entry.name for entry in resolved]
    if len(set(names)) < len(names):
        raise ValueError(
            f"Error in sktime tuner, scoring metric names must be unique, "
            f"but found {names}"
        )

    # evaluate keys its result columns by metric.__name__, so these must differ too
    metric_names = [entry.metric.__name__ for entry in resolved]
    if len(set(metric_names)) < len(metric_names):
        raise ValueError(
            "Error in sktime tuner, scoring metrics must resolve to distinct "
            f"metric functions, but {names} resolve to {metric_names}. "
            "To score the same metric with different arguments, pass metric "
            "callables with distinct __name__ attributes."
        )
