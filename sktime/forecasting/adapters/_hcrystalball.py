# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Adapter for using HCrystalBall forecastsers in sktime."""

__author__ = ["MichalChromcak"]

import pandas as pd
from sklearn.base import clone

from sktime.forecasting.base import BaseForecaster
from sktime.utils.dependencies import _check_soft_dependencies


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
    pd.Series : Predictions in form of series
    """
    return y_pred.iloc[:, 0]


class HCrystalBallAdapter(BaseForecaster):
    """Adapter for using ``hcrystalball`` forecasters in sktime.

    Adapter class - wraps any forecaster from ``hcrystalball``
    and allows using it as an ``sktime`` ``BaseForecaster``.

    Parameters
    ----------
    model : The HCrystalBall forecasting model to use.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": "MichalChromcak",
        "maintainers": "MichalChromcak",
        "python_dependencies": "hcrystalball",
        # estimator type
        # --------------
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
    }

    def __init__(self, model):
        self.model = model
        super().__init__()

    def _fit(self, y, X, fh):
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

    def _predict(self, fh=None, X=None):
        """Make forecasts for the given forecast horizon.

        Parameters
        ----------
        fh : int, list or np.array
            The forecast horizon with the steps ahead to predict
        X : pd.DataFrame, optional (default=None)
            Exogenous variables (ignored)

        Returns
        -------
        y_pred : pd.Series
            Point predictions for the forecast
        """
        X_pred = _get_X_pred(X, index=fh.to_absolute_index(self.cutoff))
        y_pred = self.model_.predict(X=X_pred)
        return _adapt_y_pred(y_pred)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict
        """
        if _check_soft_dependencies("hcrystalball", severity="none"):
            from hcrystalball.wrappers import HoltSmoothingWrapper

            params = {"model": HoltSmoothingWrapper()}

        else:
            params = {"model": 42}

        return params
