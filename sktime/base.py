"""
A collection of empty base estimators
"""

from sklearn.base import BaseEstimator


class BaseClassifier(BaseEstimator):
    """
    Base class for classifiers, for identification.
    """
    _estimator_type = "classifier"


class BaseRegressor(BaseEstimator):
    """
    Base class for regressors, for identification.
    """
    _estimator_type = "regressor"


class BaseTransformer(BaseEstimator):
    """
    Base class for transformers, for identification.
    """
    pass
