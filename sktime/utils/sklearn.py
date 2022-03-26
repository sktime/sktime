# -*- coding: utf-8 -*-
"""Sklearn related typing and inheritance checking utility."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

from inspect import isclass

from sklearn.base import BaseEstimator as SklearnBaseEstimator
from sklearn.base import ClassifierMixin, ClusterMixin, RegressorMixin, TransformerMixin

from sktime.base import BaseObject

__author__ = ["fkiraly"]


def is_sklearn_estimator(obj):
    """Check whether obj is an sklearn estimator.

    Parameters
    ----------
    obj : any class or object

    Returns
    -------
    is_sklearn_est : bool, whether obj is an sklearn estimator (class or instance)
    """
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
        "classifier" - supervised classifier
        "clusterer" - unsupervised clusterer
        "regressor" - supervised regressor
        "transformer" - transformer (pipeline element, feature extractor, unsupervised)

    Raises
    ------
    TypeError if obj is not an sklearn estimator, according to is_sklearn_estimator
    """
    if not isclass(obj):
        obj = type(obj)

    if not is_sklearn_estimator(obj):
        raise TypeError(f"{var_name} is not an sklearn estimator, has type {type(obj)}")

    sklearn_mixins = tuple(mixin_to_scitype.keys())

    if issubclass(obj, sklearn_mixins):
        for mx in sklearn_mixins:
            if issubclass(obj, mx):
                return mixin_to_scitype[mx]
    else:
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
