#!/usr/bin/env python3 -u
# coding: utf-8
<<<<<<< HEAD
=======
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed

__author__ = ["Markus Löning", "@big-o"]
__all__ = [
    "BaseForecaster",
    "DEFAULT_ALPHA",
    "is_forecaster"
]

from sktime.base import BaseEstimator

DEFAULT_ALPHA = 0.05


class BaseForecaster(BaseEstimator):
    """Base forecaster"""

<<<<<<< HEAD
    _estimator_type = "forecaster"

=======
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed
    def __init__(self):
        self._is_fitted = False
        super(BaseEstimator, self).__init__()

    def fit(self, y_train, fh=None, X_train=None):
        """Fit to training data.

        Parameters
        ----------
        y_train : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X_train : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored
        Returns
        -------
        self : returns an instance of self.
        """
        raise NotImplementedError("abstract method")

<<<<<<< HEAD
    def predict(self, fh=None, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
=======
    def predict(self, fh=None, X=None, return_pred_int=False,
                alpha=DEFAULT_ALPHA):
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed
        """Make forecasts

        Parameters
        ----------
        fh : int, list or np.array
        X : pd.DataFrame, optional (default=None)
        return_pred_int : bool, optional (default=False)
        alpha : float or list, optional (default=0.95)

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        y_pred_int : pd.DataFrame
            Prediction intervals
        """
        raise NotImplementedError("abstract method")

    def update(self, y_new, X_new=None, update_params=False):
        """Update fitted paramters

        Parameters
        ----------
        y_new : pd.Series
        X_new : pd.DataFrame
        update_params : bool, optional (default=False)

        Returns
        -------
        self : an instance of self
        """
        raise NotImplementedError("abstract method")

<<<<<<< HEAD
    def update_predict(self, y_test, cv=None, X_test=None, update_params=False, return_pred_int=False,
=======
    def update_predict(self, y_test, cv=None, X_test=None, update_params=False,
                       return_pred_int=False,
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed
                       alpha=DEFAULT_ALPHA):
        """Make and update predictions iteratively over the test set.

        Parameters
        ----------
        y_test : pd.Series
        cv : cross-validation generator, optional (default=None)
        X_test : pd.DataFrame, optional (default=None)
        update_params : bool, optional (default=False)
        return_pred_int : bool, optional (default=False)
        alpha : int or list of ints, optional (default=None)

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        y_pred_int : pd.DataFrame
            Prediction intervals
        """
        raise NotImplementedError("abstract method")

<<<<<<< HEAD
    def update_predict_single(self, y_new, fh=None, X=None, update_params=False, return_pred_int=False,
                              alpha=DEFAULT_ALPHA):
        # when nowcasting, X may be longer than y, X must be cut to same length as y so that same time points are
        # passed to update, the remaining time points of X are passed to predict
=======
    def update_predict_single(self, y_new, fh=None, X=None,
                              update_params=False, return_pred_int=False,
                              alpha=DEFAULT_ALPHA):
        # when nowcasting, X may be longer than y, X must be cut to same
        # length as y so that same time points are
        # passed to update, the remaining time points of X are passed to
        # predict
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed
        if X is not None or return_pred_int:
            raise NotImplementedError()

        self.update(y_new, X_new=X, update_params=update_params)
<<<<<<< HEAD
        return self.predict(fh=fh, X=X, return_pred_int=return_pred_int, alpha=alpha)
=======
        return self.predict(fh=fh, X=X, return_pred_int=return_pred_int,
                            alpha=alpha)
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed

    def score(self, y_test, fh=None, X=None):
        """Compute the sMAPE loss for the given forecasting horizon.

        Parameters
        ----------
        y_test : pd.Series
            Target time series to which to compare the forecasts.
        fh : int, list or array-like, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, shape=[n_obs, n_vars], optional (default=None)
            An optional 2-d dataframe of exogenous variables.

        Returns
        -------
        score : float
            sMAPE loss of self.predict(fh=fh, X=X) with respect to y_test.

        See Also
        --------
        :meth:`sktime.performance_metrics.forecasting.smape_loss`.`
        """
        # no input checks needed here, they will be performed
        # in predict and loss function
        from sktime.performance_metrics.forecasting import smape_loss
        return smape_loss(y_test, self.predict(fh=fh, X=X))

    def get_fitted_params(self):
        """Get fitted parameters

        Returns
        -------
        fitted_params : dict
        """
        raise NotImplementedError("abstract method")


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
<<<<<<< HEAD
    return getattr(estimator, "_estimator_type", None) == "forecaster"
=======
    return isinstance(estimator, BaseForecaster)
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed
