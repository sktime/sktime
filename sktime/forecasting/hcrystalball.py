# -*- coding: utf-8 -*-
# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements wrapper for using HCrystalBall forecastsers in sktime."""

import pandas as pd
from sklearn.base import clone

from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.base import BaseForecaster
from sktime.utils.validation._dependencies import _check_soft_dependencies

_check_soft_dependencies("hcrystalball")


def _check_fh(fh, cutoff):
    if fh is not None:
        if not fh.is_all_out_of_sample(cutoff):
            raise NotImplementedError(
                "in-sample prediction are currently not implemented"
            )


def _check_index(index):
    if not isinstance(index, pd.DatetimeIndex):
        raise NotImplementedError(
            "`HCrystalBallForecaster` currently only supports `pd.DatetimeIndex`. "
            "Please convert the data index to `pd.DatetimeIndex`."
        )
    return index


def _adapt_y_X(y, X):
    """Adapt fit data to HCB compliant format.

    Parameters
    ----------
    y : pd.Series
        Target variable
    X : pd.Series, pd.DataFrame
        Exogenous variables

    Returns
    -------
    tuple
        y_train - pd.Series with datetime index
        X_train - pd.DataFrame with datetime index
                  and optionally exogenous variables in columns

    Raises
    ------
    ValueError
        When neither of the argument has Datetime or Period index
    """
    index = _check_index(y.index)
    X = pd.DataFrame(index=index) if X is None else X
    return y, X


def _get_X_pred(X_pred, index):
    """Translate forecast horizon interface to HCB native dataframe.

    Parameters
    ----------
    X_pred : pd.DataFrame
        Exogenous data for predictions
    index : pd.DatetimeIndex
        Index generated from the forecasting horizon

    Returns
    -------
    pd.DataFrame
        index - datetime
        columns - exogenous variables (optional)
    """
    if X_pred is not None:
        _check_index(X_pred.index)

    X_pred = pd.DataFrame(index=index) if X_pred is None else X_pred
    return X_pred


def _adapt_y_pred(y_pred):
    """Translate wrapper prediction to sktime format.

    From Dataframe to series.

    Parameters
    ----------
    y_pred : pd.DataFrame

    Returns
    -------
    pd.Series
        Predictions in form of series
    """
    return y_pred.iloc[:, 0]


class HCrystalBallForecaster(BaseForecaster):
    """Implement wrapper to allow use of HCrystalBall forecasters in sktime.

    Parameters
    ----------
    model :
        The HCrystalBall forecasting model to use.
    """

    _tags = {
        "univariate-only": True,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
    }

    def __init__(self, model):
        self.model = model
        super(HCrystalBallForecaster, self).__init__()

    def _fit(self, y, X=None, fh=None):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series with which to fit the forecaster.
        fh : int, list or np.array, optional (default=None)
            The forecast horizon with the steps ahead to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored

        Returns
        -------
        self : returns an instance of self.
        """
        y, X = _adapt_y_X(y, X)
        self.model_ = clone(self.model)
        self.model_.fit(X, y)

        return self

    def _predict(self, fh=None, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        """Make forecasts for the given forecast horizon.

        Parameters
        ----------
        fh : int, list or np.array
            The forecast horizon with the steps ahead to predict
        X : pd.DataFrame, optional (default=None)
            Exogenous variables (ignored)
        return_pred_int : bool, optional (default=False)
            Return the prediction intervals for the forecast.
        alpha : float or list, optional (default=0.95)
            If alpha is iterable, multiple intervals will be calculated.

        Returns
        -------
        y_pred : pd.Series
            Point predictions for the forecast
        y_pred_int : pd.DataFrame
        """
        X_pred = _get_X_pred(X, index=fh.to_absolute(self.cutoff).to_pandas())
        y_pred = self.model_.predict(X=X_pred)
        return _adapt_y_pred(y_pred)

    def get_fitted_params(self):
        """Get fitted parameters."""
        raise NotImplementedError()

    def _compute_pred_err(self, alphas):
        raise NotImplementedError()
