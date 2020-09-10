__all__ = ["AutoETS"]
__author__ = ["Hongyi Yang"]

from sktime.forecasting.base._statsmodels import _StatsModelsAdapter
from statsmodels.tsa.exponential_smoothing.ets import ETSModel as _ETSModel
import numpy as np


class AutoETS(_StatsModelsAdapter):

    def __init__(
        self,
        error="add",
        trend=None,
        damped=False,
        seasonal=None,
        sp=None,
        initialization_method="estimated",
        initial_level=None,
        initial_trend=None,
        initial_seasonal=None,
        bounds=None,
        dates=None,
        freq=None,
        missing="none",
        start_params=None,
        maxiter=1000,
        full_output=True,
        disp=True,
        callback=None,
        return_params=False,
        information_criterion='aic',
        autofitting=True,
        allow_multiplicative_trend=False,
        restrict=True,
        additive_only=False,
        **kwargs
    ):

        # Model params
        self.error = error
        self.trend = trend
        self.damped = damped
        self.seasonal = seasonal
        self.sp = sp
        self.initialization_method = initialization_method
        self.initial_level = initial_level
        self.initial_trend = initial_trend
        self.initial_seasonal = initial_seasonal
        self.bounds = bounds
        self.dates = dates
        self.freq = freq
        self.missing = missing

        # Fit params
        self.start_params = start_params
        self.maxiter = maxiter
        self.full_output = full_output
        self.disp = disp
        self.callback = callback
        self.return_params = return_params
        self.information_criterion = information_criterion
        self.autofitting = autofitting
        self.allow_multiplicative_trend = allow_multiplicative_trend
        self.restrict = restrict
        self.additive_only = additive_only

        super(AutoETS, self).__init__()

    def _fit_forecaster(self, y, X_train=None):

        if self.autofitting:
            best_forecaster = None
            best_fitted_forecaster = None
            best_information_criterion = np.inf
            error_range = ['add', 'mul']
            if self.allow_multiplicative_trend:
                trend_range = ['add', 'mul', None]
            else:
                trend_range = ['add', None]
            seasonal_range = ['add', 'mul', None]
            damped_range = [True, False]

            for error in error_range:
                for trend in trend_range:
                    for seasonal in seasonal_range:
                        for damped in damped_range:

                            if trend is None and damped:
                                continue

                            if self.restrict:
                                if error == 'add' and (trend == 'mul' or
                                                       seasonal == 'mul'):
                                    continue
                                if error == 'mul' and trend == 'mul' and \
                                        seasonal == 'add':
                                    continue
                                if self.additive_only and (error == 'mul' or
                                                           trend == 'mul' or
                                                           seasonal == 'mul'):
                                    continue
                            _forecaster = _ETSModel(
                                y,
                                error=error,
                                trend=trend,
                                damped_trend=damped,
                                seasonal=seasonal,
                                seasonal_periods=self.sp,
                                initialization_method=self.
                                initialization_method,
                                initial_level=self.initial_level,
                                initial_trend=self.initial_trend,
                                initial_seasonal=self.initial_seasonal,
                                bounds=self.bounds,
                                dates=self.dates,
                                freq=self.freq,
                                missing=self.missing
                            )
                            _fitted_forecaster = _forecaster.fit(
                                start_params=self.start_params,
                                maxiter=self.maxiter,
                                full_output=self.full_output,
                                disp=self.disp,
                                callback=self.callback,
                                return_params=self.return_params
                            )

                            if self.information_criterion == 'aic':
                                _ic = _fitted_forecaster.aic
                            elif self.information_criterion == 'bic':
                                _ic = _fitted_forecaster.bic
                            elif self.information_criterion == 'aicc':
                                _ic = _fitted_forecaster.aicc
                            else:
                                raise ValueError('information criterion must \
                                                 either be aic, bic or aicc')

                            print(_ic)
                            print(best_information_criterion)
                            if _ic < best_information_criterion:
                                best_information_criterion = _ic
                                best_forecaster = _forecaster
                                best_fitted_forecaster = _fitted_forecaster

            self._forecaster = best_forecaster
            self._fitted_forecaster = best_fitted_forecaster

        else:
            self._forecaster = _ETSModel(
                y,
                error=self.error,
                trend=self.trend,
                damped_trend=self.damped,
                seasonal=self.seasonal,
                seasonal_periods=self.sp,
                initialization_method=self.initialization_method,
                initial_level=self.initial_level,
                initial_trend=self.initial_trend,
                initial_seasonal=self.initial_seasonal,
                bounds=self.bounds,
                dates=self.dates,
                freq=self.freq,
                missing=self.missing
            )

            self._fitted_forecaster = self._forecaster.fit(
                start_params=self.start_params,
                maxiter=self.maxiter,
                full_output=self.full_output,
                disp=self.disp,
                callback=self.callback,
                return_params=self.return_params
            )

    def summary(self):
        return self._fitted_forecaster.summary()
