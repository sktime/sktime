"""Sklearn related typing and inheritance checking utility."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

from inspect import isclass, signature

from sklearn.base import BaseEstimator as SklearnBaseEstimator
from sklearn.base import ClassifierMixin, ClusterMixin, RegressorMixin, TransformerMixin
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

__author__ = ["fkiraly"]


def is_in_sklearn(obj):
    """Check whether obj is in sklearn.

    Parameters
    ----------
    obj : any class or object

    Returns
    -------
    is_in_sklearn : bool, whether obj is in sklearn (class or instance)
    """
    mod = getattr(obj, "__module__", "")
    if not (mod.startswith("sklearn.") or mod == "sklearn"):
        return False

    return True


def is_sklearn_object(obj):
    """Check whether obj is an sklearn object.

    Parameters
    ----------
    obj : any class or object

    Returns
    -------
    is_sklearn_obj : bool, whether obj is an sklearn object (class or instance)
    """
    res = is_sklearn_estimator(obj)
    res = res or is_sklearn_metric(obj)
    res = res or is_sklearn_splitter(obj)
    return res


def is_sklearn_estimator(obj):
    """Check whether obj is an sklearn estimator.

    Parameters
    ----------
    obj : any class or object

    Returns
    -------
    is_sklearn_est : bool, whether obj is an sklearn estimator (class or instance)
    """
    from sktime.base import BaseObject

    if not isclass(obj):
        obj = type(obj)

    is_in_sklearn = issubclass(obj, SklearnBaseEstimator)
    is_in_sktime = issubclass(obj, BaseObject)

    is_sklearn_est = is_in_sklearn and not is_in_sktime
    return is_sklearn_est


mixin_to_scitype = {
    ClassifierMixin: "classifier",
    ClusterMixin: "clusterer",
    RegressorMixin: "regressor",
    TransformerMixin: "transformer",
}


def sklearn_scitype(obj, var_name="obj"):
    """Return sklearn scitype.

    Parameters
    ----------
    obj : any class or object
    var_name : str, optional, default = "obj"
        name of variable (obj) to display in error message

    Returns
    -------
    str, the sklearn scitype of obj, inferred from inheritance tree, one of

    * "classifier" - supervised classifier
    * "clusterer" - unsupervised clusterer
    * "metric" - sklearn metric function
    * "regressor" - supervised regressor
    * "splitter" - sklearn splitter (cross-validation generator)
    * "transformer" - transformer (pipeline element, feature extractor, unsupervised)
    * "estimator" - sklearn estimator of indeterminate type

    Raises
    ------
    TypeError if obj is not an sklearn estimator, according to is_sklearn_estimator
    """
    is_metric, metric_type = is_sklearn_metric(obj, return_type=True)
    if is_metric:
        return metric_type

    if is_sklearn_splitter(obj):
        return "splitter"

    if not is_sklearn_estimator(obj):
        raise TypeError(f"{var_name} is not an sklearn estimator, has type {type(obj)}")

    # first check whether obj class inherits from sklearn mixins
    sklearn_mixins = tuple(mixin_to_scitype.keys())

    if not isclass(obj):
        obj_class = type(obj)
    else:
        obj_class = obj
    if issubclass(obj_class, sklearn_mixins):
        for mx in sklearn_mixins:
            if issubclass(obj_class, mx):
                return mixin_to_scitype[mx]

    # deal with sklearn pipelines: scitype is determined by the last element
    if isinstance(obj, Pipeline) or hasattr(obj, "steps"):
        return sklearn_scitype(obj.steps[-1][1], var_name=var_name)

    # deal with generic composites: scitype is type of wrapped "estimator"
    if isinstance(obj, (GridSearchCV, RandomizedSearchCV)) or hasattr(obj, "estimator"):
        return sklearn_scitype(obj.estimator, var_name=var_name)

    # fallback - estimator of indeterminate type
    return "estimator"


def is_sklearn_transformer(obj):
    """Check whether obj is an sklearn transformer.

    Parameters
    ----------
    obj : any object

    Returns
    -------
    bool, whether obj is an sklearn transformer
    """
    return is_sklearn_estimator(obj) and sklearn_scitype(obj) == "transformer"


def is_sklearn_classifier(obj):
    """Check whether obj is an sklearn classifier.

    Parameters
    ----------
    obj : any object

    Returns
    -------
    bool, whether obj is an sklearn classifier
    """
    return is_sklearn_estimator(obj) and sklearn_scitype(obj) == "classifier"


def is_sklearn_regressor(obj):
    """Check whether obj is an sklearn regressor.

    Parameters
    ----------
    obj : any object

    Returns
    -------
    bool, whether obj is an sklearn regressor
    """
    return is_sklearn_estimator(obj) and sklearn_scitype(obj) == "regressor"


def is_sklearn_clusterer(obj):
    """Check whether obj is an sklearn clusterer.

    Parameters
    ----------
    obj : any object

    Returns
    -------
    bool, whether obj is an sklearn clusterer
    """
    return is_sklearn_estimator(obj) and sklearn_scitype(obj) == "clusterer"


def is_sklearn_splitter(obj):
    """Check whether obj is an sklearn splitter.

    Check whether a class conforms to the sklearn splitter API signature.
    Does not check input/output contract.

    Conditions:

    - Has a callable ``split(X, y=None, groups=None)`` method
    - Has a callable ``get_n_splits(X=None, y=None, groups=None)`` method
    - splitter import location is in ``sklearn`` module

    Parameters
    ----------
    obj : any object

    Returns
    -------
    bool, whether obj is an sklearn splitter
    """
    from skbase.base import BaseObject

    # if obj is a sktime BaseObject, return False right away
    if not is_in_sklearn(obj) or issubclass(type(obj), BaseObject):
        return False

    # Instantiate if a class is passed
    try:
        obj = obj() if isclass(obj) else obj
    except Exception:
        return False

    # 1. Check `split`
    if not hasattr(obj, "split") or not callable(getattr(obj, "split")):
        return False
    split_sig = signature(obj.split)
    split_params = list(split_sig.parameters.keys())
    if len(split_params) < 1 or split_params[0] != "X":
        return False

    # 2. Check `get_n_splits`
    if not hasattr(obj, "get_n_splits") or not callable(getattr(obj, "get_n_splits")):
        return False
    gns_sig = signature(obj.get_n_splits)
    gns_params = list(gns_sig.parameters.keys())
    if len(gns_params) > 3:
        return False

    return True


def is_sklearn_metric(obj, return_type=False):
    """Check whether obj is an sklearn metric.

    Check whether an object conforms to sklearn's metric API signature.
    Does not check input/output contract.

    Conditions:

    - Callable
    - Signature: (y_true, y_pred, ...)
    - splitter import location is in ``sklearn`` module

    Parameters
    ----------
    obj : any object
    return_type : bool, optional, default=False
        whether to return the type of metric if obj is a metric

    Returns
    -------
    bool : whether obj is an sklearn metric
    type : str, one of

        * "metric": non-probabilistic metric for regression or classification,
          of signature (y_true, y_pred, ...)
        * "metric_proba": metric that takes probability estimates as second argument,
          of signature (y_true, y_proba, ...)
        * None: if obj is not an sklearn metric
    """

    def _ret(res, typ):
        if return_type:
            return res, typ
        return res

    from skbase.base import BaseObject

    # if obj is a sktime BaseObject, return False right away
    if not is_in_sklearn(obj) or issubclass(type(obj), BaseObject):
        return _ret(False, None)

    # 1. Must be callable
    if not callable(obj):
        return _ret(False, None)

    # 2. Must be from sklearn.*
    mod = getattr(obj, "__module__", "")
    if not (mod.startswith("sklearn.") or mod == "sklearn"):
        return _ret(False, None)

    # 3. Must have correct signature
    try:
        sig = signature(obj)
    except (TypeError, ValueError):
        return _ret(False, None)

    params = list(sig.parameters.values())
    if len(params) < 2:
        return _ret(False, None)
    if params[0].name != "y_true":
        return _ret(False, None)
    # deterministic metrics have signature (y_true, y_pred, ...)
    # probabilistic metrics have signature (y_true, y_proba, ...)
    # earlier versions of sklearn sometimes use "y_prob" instead of "y_proba"
    if params[1].name not in {"y_pred", "y_proba", "y_prob"}:
        return _ret(False, None)

    if params[1].name == "y_pred":
        return _ret(True, "metric")
    if params[1].name in ["y_proba", "y_prob"]:
        return _ret(True, "metric_proba")

    # unreachable due to "not in" above, but for code checkers
    _ret(False, None)
