from sklearn.base import BaseEstimator


class BaseRegressor(BaseEstimator):
    """
    Base class for regressors, for identification.
    """
    _estimator_type = "regressor"