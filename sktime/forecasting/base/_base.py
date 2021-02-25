#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning", "@big-o"]
__all__ = ["BaseForecaster"]

from sktime.base import BaseEstimator


DEFAULT_ALPHA = 0.05


class BaseForecaster(BaseEstimator):
    """Base forecaster

    The base forecaster specifies the methods and method
    signatures that all forecasters have to implement.

    Specific implementations of these methods is deferred to concrete
    forecasters.
    """

    def __init__(self):
        self._is_fitted = False
        super(BaseForecaster, self).__init__()

    def fit(self, y, X=None, fh=None):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored
        Returns
        -------
        self : returns an instance of self.
        """
        raise NotImplementedError("abstract method")

    def predict(self, fh=None, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        """Make forecasts

        Parameters
        ----------
        fh : int, list or np.array
        X : pd.DataFrame, optional (default=None)
        return_pred_int : bool, optional (default=False)
        alpha : float or list, optional (default=0.95)
            A significance level or list of significance levels.

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        y_pred_int : pd.DataFrame
            Prediction intervals
        """
        raise NotImplementedError("abstract method")

    def compute_pred_int(self, y_pred, alpha=DEFAULT_ALPHA):
        """
        Get the prediction intervals for a forecast.

        If alpha is iterable, multiple intervals will be calculated.

        Parameters
        ----------

        y_pred : pd.Series
            Point predictions.

        alpha : float or list, optional (default=0.95)
            A significance level or list of significance levels.

        Returns
        -------

        intervals : pd.DataFrame
            A table of upper and lower bounds for each point prediction in
            ``y_pred``. If ``alpha`` was iterable, then ``intervals`` will be a
            list of such tables.
        """
        raise NotImplementedError("abstract method")

    def update(self, y, X=None, update_params=True):
        """Update fitted parameters

        Parameters
        ----------
        y : pd.Series
        X : pd.DataFrame
        update_params : bool, optional (default=True)

        Returns
        -------
        self : an instance of self
        """
        raise NotImplementedError("abstract method")

    def update_predict(
        self,
        y,
        cv=None,
        X=None,
        update_params=True,
        return_pred_int=False,
        alpha=DEFAULT_ALPHA,
    ):
        """Make and update predictions iteratively over the test set.

        Parameters
        ----------
        y : pd.Series
        cv : cross-validation generator, optional (default=None)
        X : pd.DataFrame, optional (default=None)
        update_params : bool, optional (default=True)
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

    def update_predict_single(
        self,
        y_new,
        fh=None,
        X=None,
        update_params=True,
        return_pred_int=False,
        alpha=DEFAULT_ALPHA,
    ):
        # when nowcasting, X may be longer than y, X must be cut to same
        # length as y so that same time points are
        # passed to update, the remaining time points of X are passed to
        # predict
        if X is not None or return_pred_int:
            raise NotImplementedError()

        self.update(y_new, X, update_params=update_params)
        return self.predict(fh, X, return_pred_int=return_pred_int, alpha=alpha)

    def score(self, y, X=None, fh=None):
        """Compute the sMAPE loss for the given forecasting horizon.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to compare the forecasts.
        fh : int, list or array-like, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, shape=[n_obs, n_vars], optional (default=None)
            An optional 2-d dataframe of exogenous variables.

        Returns
        -------
        score : float
            sMAPE loss of self.predict(fh, X) with respect to y_test.

        See Also
        --------
        :meth:`sktime.performance_metrics.forecasting.smape_loss`.`
        """
        # no input checks needed here, they will be performed
        # in predict and loss function
        from sktime.performance_metrics.forecasting import smape_loss

        return smape_loss(y, self.predict(fh, X))

    def get_fitted_params(self):
        """Get fitted parameters

        Returns
        -------
        fitted_params : dict
        """
        raise NotImplementedError("abstract method")
