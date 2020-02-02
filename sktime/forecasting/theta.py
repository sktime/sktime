import numpy as np
import pandas as pd

from sktime.forecasting import ExpSmoothingForecaster
from sktime.forecasting.base import DEFAULT_ALPHA
from sktime.transformers.forecasting import Deseasonaliser
from sktime.utils.confidence import zscore
from sktime.utils.validation.forecasting import check_alpha, validate_sp, validate_y
from sktime.utils.seasonality import seasonality_test
from sktime.utils.time_series import fit_trend

__all__ = ["ThetaForecaster"]
__author__ = ["@big-o"]


class ThetaForecaster(ExpSmoothingForecaster):
    """
    Theta method of forecasting.

    The theta method as defined in [1]_ is equivalent to simple exponential smoothing
    (SES) with drift. This is demonstrated in [2]_.

    The series is tested for seasonality using the test outlined in A&N. If deemed
    seasonal, the series is seasonally adjusted using a classical multiplicative
    decomposition before applying the theta method. The resulting forecasts are then
    reseasonalized.

    In cases where SES results in a constant forecast, the theta forecaster will revert
    to predicting the SES constant plus a linear trend derived from the training data.

    Prediction intervals are computed using the underlying state space model.

    Parameters
    ----------

    smoothing_level : float, optional
        The alpha value of the simple exponential smoothing, if the value is set then
        this will be used, otherwise it will be estimated from the data.

    deseasonaliser : :class:`sktime.transformers.Deseasonaliser`, optional (default=None)
        A transformer to use for seasonal adjustments. Overrides the
        ``seasonal_periods`` parameter.

    seasonal_periods : int, optional (default=1)
        The number of observations that constitute a seasonal period for a
        multiplicative deseasonaliser, which is used if seasonality is detected in the
        training data. Ignored if a deseasonaliser transformer is provided. Default is
        1 (no seasonality).

    Attributes
    ----------

    smoothing_level_ : float
        The estimated alpha value of the SES fit.

    drift_ : float
        The estimated drift of the fitted model.

    se_ : float
        The standard error of the predictions. Used to calculate prediction intervals.

    References
    ----------

    .. [1] `Assimakopoulos, V. and Nikolopoulos, K. The theta model: a decomposition
           approach to forecasting. International Journal of Forecasting 16, 521-530,
           2000.
           <https://www.sciencedirect.com/science/article/pii/S0169207000000662>`_

    .. [2] `Hyndman, Rob J., and Billah, Baki. Unmasking the Theta method.
           International J. Forecasting, 19, 287-290, 2003.
           <https://www.sciencedirect.com/science/article/pii/S0169207001001431>`_
    """

    def __init__(self, smoothing_level=None, deseasonaliser=None, seasonal_periods=1):
        self.deseasonaliser = deseasonaliser
        self.seasonal_periods = seasonal_periods
        self.smoothing_level = smoothing_level

        if deseasonaliser:
            self._deseasonaliser = deseasonaliser
        elif seasonal_periods is not None:
            self._deseasonaliser = Deseasonaliser(
                model="multiplicative", sp=seasonal_periods
            )
        else:
            raise ValueError(
                "One of 'seasonal_periods' or 'deseasonaliser' must be provided."
            )

        self.trend_ = None
        self.smoothing_level_ = None
        super(ThetaForecaster, self).__init__(smoothing_level=smoothing_level)

    def _to_nested(self, y):
        nested = pd.DataFrame(pd.Series([y]))

        return nested

    def fit(self, y_train, fh=None, X_train=None):
        """
        Fit to training data.

        Parameters
        ----------

        y_train : pandas.Series
            Target time series to which to fit the forecaster.
        fh : array-like, optional (default=[1])
            The forecasters horizon with the steps ahead to to predict.

        Returns
        -------

        self : returns an instance of self.
        """

        y_train = validate_y(y_train)
        fh = self._set_fh(fh)

        # update observation horizon
        self._set_obs_horizon(y_train.index)

        y_train = self._deseasonalise(y_train)

        # Find theta lines. Theta lines are just SES + drift.
        super().fit(y_train, fh=fh)
        self.smoothing_level_ = self._fitted_estimator.params["smoothing_level"]
        self.trend_ = self._compute_trend(y_train)

        self._is_fitted = True

        return self

    def _deseasonalise(self, y):
        if self._deseasonaliser.sp == 1:
            self._is_seasonal = False
        else:
            self._is_seasonal = seasonality_test(y, freq=self._deseasonaliser.sp)

        if self._is_seasonal:
            y_nested = self._to_nested(y)
            y = self._deseasonaliser.fit_transform(y_nested).iloc[0, 0]

        return y

    def predict(self, fh=None, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        """
        Make forecasts.

        Parameters
        ----------

        fh : array-like
            The forecasters horizon with the steps ahead to to predict. Default is
            one-step ahead forecast, i.e. np.array([1]).

        Returns
        -------

        y_pred : pandas.Series
            Returns series of predicted values.
        """

        self._check_is_fitted()

        # Set forecast horizon.
        self._set_fh(fh)

        # SES.
        y_pred = super(ThetaForecaster, self).predict()

        # Add drift.
        drift = self._compute_drift()
        y_pred += drift

        if self._is_seasonal:
            # Reseasonalise.
            y_pred_nested = self._to_nested(y_pred)
            y_pred = self._deseasonaliser.inverse_transform(y_pred_nested).iloc[0, 0]

        if return_pred_int:
            intvl = self.compute_pred_int(y_pred=y_pred, alpha=alpha)
            return y_pred, intvl

        return y_pred

    def _compute_trend(self, y):
        # Trend calculated through least squares regression.
        coefs = fit_trend(y.values.reshape(1, -1), order=1)
        return coefs[0, 0] / 2

    def _compute_drift(self):
        if np.isclose(self.smoothing_level_, 0.0):
            # SES was constant so revert to simple trend
            drift = self.trend_ * self.fh
        else:
            # Calculate drift from SES parameters
            n_obs = len(self._obs_horizon)
            drift = self.trend_ * (
                self.fh
                + (1 - (1 - self.smoothing_level_) ** n_obs) / self.smoothing_level_
            )

        return drift

    def compute_pred_errs(self, alpha=DEFAULT_ALPHA):
        """
        Get the prediction errors for the forecast.
        """
        self._check_is_fitted()
        check_alpha(alpha)

        n_obs = len(self._obs_horizon)

        self.sigma_ = np.sqrt(self._fitted_estimator.sse / (n_obs - 1))
        sem = self.sigma_ * np.sqrt(self._fh * self.smoothing_level_ ** 2 + 1)

        if isinstance(alpha, (np.integer, np.float)):
            z = zscore(1 - alpha)
            err = z * sem

            return pd.Series(index=self._get_absolute_fh(), data=err)

        errs = []
        for al in alpha:
            z = zscore(1 - al)
            err = z * sem
            errs.append(pd.Series(index=self._get_absolute_fh(), data=err))

        return tuple(errs)

    def update(self, y_new, X_new=None, update_params=True):
        # update observation horizon
        super(ThetaForecaster, self).update(
            y_new, X_new=None, update_params=update_params
        )

        if update_params:
            y_new = self._deseasonalise(y_new)
            self.smoothing_level_ = self._fitted_estimator.params["smoothing_level"]
            self.trend_ = self._compute_trend(y_new)
