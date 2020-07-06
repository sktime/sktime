__all__ = ["ThetaForecaster"]
__author__ = ["@big-o", "Markus Löning"]

from warnings import warn

import numpy as np
import pandas as pd
from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.transformers.single_series.detrend import Deseasonalizer
from sktime.utils.confidence import zscore
from sktime.utils.time_series import fit_trend
from sktime.utils.validation.forecasting import check_alpha
from sktime.utils.validation.forecasting import check_sp


class ThetaForecaster(ExponentialSmoothing):
    """
    Theta method of forecasting.

    The theta method as defined in [1]_ is equivalent to simple exponential
    smoothing
    (SES) with drift. This is demonstrated in [2]_.

    The series is tested for seasonality using the test outlined in A&N. If
    deemed
    seasonal, the series is seasonally adjusted using a classical
    multiplicative
    decomposition before applying the theta method. The resulting forecasts
    are then
    reseasonalized.

    In cases where SES results in a constant forecast, the theta forecaster
    will revert
    to predicting the SES constant plus a linear trend derived from the
    training data.

    Prediction intervals are computed using the underlying state space model.

    Parameters
    ----------

    smoothing_level : float, optional
        The alpha value of the simple exponential smoothing, if the value is
        set then
        this will be used, otherwise it will be estimated from the data.

    deseasonalise : bool, optional (default=True)
        If True, data is seasonally adjusted.

    sp : int, optional (default=1)
        The number of observations that constitute a seasonal period for a
        multiplicative deseasonaliser, which is used if seasonality is
        detected in the
        training data. Ignored if a deseasonaliser transformer is provided.
        Default is
        1 (no seasonality).

    Attributes
    ----------

    smoothing_level_ : float
        The estimated alpha value of the SES fit.

    drift_ : float
        The estimated drift of the fitted model.

    se_ : float
        The standard error of the predictions. Used to calculate prediction
        intervals.

    References
    ----------

    .. [1] `Assimakopoulos, V. and Nikolopoulos, K. The theta model: a
    decomposition
           approach to forecasting. International Journal of Forecasting 16,
           521-530,
           2000.
           <https://www.sciencedirect.com/science/article/pii
           /S0169207000000662>`_

    .. [2] `Hyndman, Rob J., and Billah, Baki. Unmasking the Theta method.
           International J. Forecasting, 19, 287-290, 2003.
           <https://www.sciencedirect.com/science/article/pii
           /S0169207001001431>`_
    """

    _fitted_param_names = ("initial_level", "smoothing_level")

    def __init__(self, smoothing_level=None, deseasonalise=True, sp=1):

        self.sp = sp
        self.deseasonalise = deseasonalise

        self.deseasonaliser_ = None
        self.trend_ = None
        self.smoothing_level_ = None
        self.drift_ = None
        self.se_ = None
        super(ThetaForecaster, self).__init__(smoothing_level=smoothing_level,
                                              sp=sp)

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
        sp = check_sp(self.sp)
        if sp > 1 and not self.deseasonalise:
            warn("`sp` is ignored when `deseasonalise`=False")

        if self.deseasonalise:
            self.deseasonaliser_ = Deseasonalizer(sp=self.sp,
                                                  model="multiplicative")
            y_train = self.deseasonaliser_.fit_transform(y_train)

        # fit exponential smoothing forecaster
        # find theta lines: Theta lines are just SES + drift
        super(ThetaForecaster, self).fit(y_train, fh=fh)
        self.smoothing_level_ = self._fitted_forecaster.params[
            "smoothing_level"]

        # compute trend
        self.trend_ = self._compute_trend(y_train)
        self._is_fitted = True
        return self

    def _predict(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        """
        Make forecasts.

        Parameters
        ----------

        fh : array-like
            The forecasters horizon with the steps ahead to to predict.
            Default is
            one-step ahead forecast, i.e. np.array([1]).

        Returns
        -------

        y_pred : pandas.Series
            Returns series of predicted values.
        """
        y_pred = super(ThetaForecaster, self)._predict(fh, X=X,
                                                       return_pred_int=False,
                                                       alpha=alpha)

        # Add drift.
        drift = self._compute_drift()
        y_pred += drift

        if self.deseasonalise:
            y_pred = self.deseasonaliser_.inverse_transform(y_pred)

        if return_pred_int:
            pred_int = self.compute_pred_int(y_pred=y_pred, alpha=alpha)
            return y_pred, pred_int

        return y_pred

    @staticmethod
    def _compute_trend(y):
        # Trend calculated through least squares regression.
        coefs = fit_trend(y.values.reshape(1, -1), order=1)
        return coefs[0, 0] / 2

    def _compute_drift(self):
        if np.isclose(self.smoothing_level_, 0.0):
            # SES was constant, so revert to simple trend
            drift = self.trend_ * self.fh
        else:
            # Calculate drift from SES parameters
            n_timepoints = len(self.oh)
            drift = self.trend_ * (
                    self.fh
                    + (1 - (
                        1 - self.smoothing_level_) ** n_timepoints) /
                    self.smoothing_level_
            )

        return drift

    def _compute_pred_errors(self, alpha=DEFAULT_ALPHA):
        """
        Get the prediction errors for the forecast.
        """
        self.check_is_fitted()
        alpha = check_alpha(alpha)

        n_timepoints = len(self.oh)

        self.sigma_ = np.sqrt(self._fitted_forecaster.sse / (n_timepoints - 1))
        sem = self.sigma_ * np.sqrt(self._fh * self.smoothing_level_ ** 2 + 1)

        errors = []
        for a in alpha:
            z = zscore(1 - a)
            error = z * sem
            errors.append(
                pd.Series(error, index=self.fh.absolute(self.cutoff)))

        # for a single alpha value, unwrap list
        if len(errors) == 1:
            return errors[0]

        # otherwise, return list of errors
        return errors

    def update(self, y_new, X_new=None, update_params=True):
        super(ThetaForecaster, self).update(y_new, X_new=X_new,
                                            update_params=update_params)
        if update_params:
            if self.deseasonalise:
                y_new = self.deseasonaliser_.transform(y_new)
            self.smoothing_level_ = self._fitted_forecaster.params[
                "smoothing_level"]
            self.trend_ = self._compute_trend(y_new)
        return self

    def compute_pred_int(self, y_pred, alpha=DEFAULT_ALPHA):
        """
        Get the prediction intervals for the forecast. If alpha is iterable,
        multiple
        intervals will be calculated.
        """
        errors = self._compute_pred_errors(alpha=alpha)

        # for multiple alphas, errors come in a list;
        # for single alpha, they come as a single pd.Series,
        # wrap it here into a list to make it iterable,
        # to avoid code duplication
        if isinstance(errors, pd.Series):
            errors = [errors]

        # compute prediction intervals
        pred_int = [
            pd.DataFrame({
                "lower": y_pred - error,
                "upper": y_pred + error
            })
            for error in errors
        ]

        # for a single alpha, return single pd.DataFrame
        if len(pred_int) == 1:
            return pred_int[0]

        # otherwise return list of pd.DataFrames
        return pred_int
