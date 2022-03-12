# -*- coding: utf-8 -*-
"""Sklearn related typing and inheritance checking utility."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

from sklearn.base import BaseEstimator as SklearnBaseEstimator
from sklearn.base import ClassifierMixin, ClusterMixin, RegressorMixin, TransformerMixin

from sktime.base import BaseObject

__author__ = ["fkiraly"]


def is_sklearn_estimator(obj):
    """Check whether obj is an sklearn estimator.

    Parameters
    ----------
    obj : any object

    Returns
    -------
    is_sklearn_est : bool, whether obj is an sklearn estimator
    """
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
    obj : any object
    var_name : str, optional, default = "obj"
        name of variable (obj) to display in error message

    Returns
    -------
    str, sklearn scitype inferred, one of
        "classifier" - supervised classifier
        "clusterer" - unsupervised clusterer
        "regressor" - supervised regressor
        "transformer" - transformer (pipeline element, feature extractor, unsupervised)

    Raises
    ------
    TypeError if obj is not an sklearn estimator, according to is_sklearn_estimator
    """
    if not is_sklearn_estimator(obj):
        raise TypeError(f"{var_name} is not an sklearn estimator, has type {type(obj)}")

    sklearn_mixins = tuple(mixin_to_scitype.keys())

    if issubclass(obj, sklearn_mixins):
        for mx in sklearn_mixins:
            if issubclass(obj, mx):
                return mixin_to_scitype[mx]
    else:
        return "estimator"
