# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""
Abstract base class for early time series classifiers.

    class name: BaseEarlyClassifier

Defining methods:
    fitting                 - fit(self, X, y)
    predicting              - predict(self, X)
                            - predict_proba(self, X)
    updating predictions    - update_predict(self, X)
      (streaming)           - update_predict_proba(self, X)

Inherited inspection methods:
    hyper-parameter inspection  - get_params()
    fitted parameter inspection - get_fitted_params()

State:
    fitted model/strategy   - by convention, any attributes ending in "_"
    fitted state flag       - is_fitted (property)
    fitted state inspection - check_is_fitted()
    streaming decision info - state_info attribute
"""

__all__ = [
    "BaseEarlyClassifier",
]
__author__ = ["mloning", "fkiraly", "TonyBagnall", "MatthewMiddlehurst"]

from deprecated.sphinx import deprecated

from sktime.classification.early_classification.base import (
    BaseEarlyClassifier as new_base,
)


# TODO: remove message in v0.15.0 and change base class
@deprecated(
    version="0.13.4",
    reason="BaseEarlyClassifier has moved and this import will be removed in 1.15.0. Import from sktime.classification.early_classification.base",  # noqa: E501
    category=FutureWarning,
)
class BaseEarlyClassifier(new_base):
    """Abstract base class for early time series classifiers.

    The base classifier specifies the methods and method signatures that all
    early classifiers have to implement. Attributes with an underscore suffix are set in
    the method fit.

    Parameters
    ----------
    classes_            : ndarray of class labels, possibly strings
    n_classes_          : integer, number of classes (length of classes_)
    fit_time_           : integer, time (in milliseconds) for fit to run.
    _class_dictionary   : dictionary mapping classes_ onto integers 0...n_classes_-1.
    _threads_to_use     : number of threads to use in fit as determined by n_jobs.
    state_info          : An array containing the state info for each decision in X.
    """

    def __init__(self):
        super(BaseEarlyClassifier, self).__init__()
