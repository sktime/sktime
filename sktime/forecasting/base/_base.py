#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning", "@big-o"]
__all__ = [
    "BaseForecaster",
    "DEFAULT_ALPHA",
    "is_forecaster"
]

from sklearn.base import BaseEstimator
from sktime.utils.exceptions import NotFittedError

DEFAULT_ALPHA = 0.05


class BaseForecaster(BaseEstimator):
    """Base forecaster"""

    _estimator_type = "forecaster"

    def __init__(self):
        self._is_fitted = False

    def fit(self, y_train, fh=None, X_train=None):
        raise NotImplementedError("abstract method")

    def predict(self, fh=None, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        raise NotImplementedError("abstract method")

    def update(self, y_new, X_new=None, update_params=False):
        raise NotImplementedError("abstract method")

    def update_predict(self, y_test, cv=None, X_test=None, update_params=False, return_pred_int=False,
                       alpha=DEFAULT_ALPHA):
        raise NotImplementedError("abstract method")

    def update_predict_single(self, y_new, fh=None, X=None, update_params=False, return_pred_int=False,
                              alpha=DEFAULT_ALPHA):
        """Allows for more efficient update-predict routines than calling them sequentially"""
        # when nowcasting, X may be longer than y, X must be cut to same length as y so that same time points are
        # passed to update, the remaining time points of X are passed to predict
        if X is not None or return_pred_int:
            raise NotImplementedError()

        self.update(y_new, X_new=X, update_params=update_params)
        return self.predict(fh=fh, X=X, return_pred_int=return_pred_int, alpha=alpha)

    def score(self, y_test, fh=None, X=None):
        """
        Returns the SMAPE on the given forecast horizon.
        Parameters
        ----------
        y_test : pandas.Series
            Target time series to which to compare the forecasts.
        fh : int or array-like, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pandas.DataFrame, shape=[n_obs, n_vars], optional (default=None)
            An optional 2-d dataframe of exogenous variables.
        Returns
        -------
        score : float
            SMAPE score of self.predict(fh=fh, X=X) with respect to y.
        See Also
        --------
        :meth:`sktime.performance_metrics.forecasting.smape_loss`.`
        """
        # no input checks needed here, they will be performed
        # in predict and loss function
        from sktime.performance_metrics.forecasting import smape_loss
        return smape_loss(y_test, self.predict(fh=fh, X=X))

    def get_fitted_params(self):
        raise NotImplementedError("abstract method")

    @property
    def is_fitted(self):
        """Has `fit` been called?"""
        return self._is_fitted

    def _check_is_fitted(self):
        """Check if the forecaster has been fitted.

        Raises
        ------
        NotFittedError
            if the forecaster has not been fitted yet.
        """
        if not self.is_fitted:
            raise NotFittedError(f"This instance of {self.__class__.__name__} has not "
                                 f"been fitted yet; please call `fit` first.")


def is_forecaster(estimator):
    """Return True if the given estimator is (probably) a forecaster.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is a forecaster and False otherwise.
    """
    return getattr(estimator, "_estimator_type", None) == "forecaster"
