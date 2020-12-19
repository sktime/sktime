# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.base import clone

from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.base._sktime import _OptionalForecastingHorizonMixin
from sktime.forecasting.base._sktime import _SktimeForecaster
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
    """Adapt fit data to HCB compliant format

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
    """Translate forecast horizon interface to HCB native dataframe

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
    """Translate wrapper prediction to sktime format

    From Dataframe to series

    Parameters
    ----------
    y_pred : pd.DataFrame

    Returns
    -------
    pd.Series
        Predictions in form of series
    """
    return y_pred.iloc[:, 0]


class HCrystalBallForecaster(_OptionalForecastingHorizonMixin, _SktimeForecaster):
    def __init__(self, model):
        self.model = model
        super(HCrystalBallForecaster, self).__init__()

    def fit(self, y, X=None, fh=None):
        self._set_y_X(y, X)
        self._set_fh(fh)

        if fh is not None:
            _check_fh(self.fh, self.cutoff)

        y, X = _adapt_y_X(y, X)
        self.model_ = clone(self.model)
        self.model_.fit(X, y)

        self._is_fitted = True
        return self

    def _predict(self, fh=None, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        if return_pred_int:
            raise NotImplementedError()
        _check_fh(fh, self.cutoff)

        X_pred = _get_X_pred(X, index=fh.to_absolute(self.cutoff).to_pandas())
        y_pred = self.model_.predict(X=X_pred)
        return _adapt_y_pred(y_pred)

    def get_fitted_params(self):
        raise NotImplementedError()

    def _compute_pred_err(self, alphas):
        raise NotImplementedError()
