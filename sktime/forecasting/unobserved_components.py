# -*- coding: utf-8 -*-

from sktime.forecasting.base.adapters import _StatsModelsAdapter
from sktime.forecasting.base._base import DEFAULT_ALPHA


class UnobservedComponents(_StatsModelsAdapter):
    def __init__(
        self,
        level=False,
        trend=False,
        seasonal=None,
        freq_seasonal=None,
        cycle=False,
        autoregressive=None,
        irregular=False,
        stochastic_level=False,
        stochastic_trend=False,
        stochastic_seasonal=True,
        stochastic_freq_seasonal=None,
        stochastic_cycle=False,
        damped_cycle=False,
        cycle_period_bounds=None,
        mle_regression=True,
        use_exact_diffuse=False,
        **kwargs
    ):

        self.level = (level,)
        self.trend = (trend,)
        self.seasonal = (seasonal,)
        self.freq_seasonal = (freq_seasonal,)
        self.cycle = (cycle,)
        self.autoregressive = (autoregressive,)
        self.irregular = (irregular,)
        self.stochastic_level = (stochastic_level,)
        self.stochastic_trend = (stochastic_trend,)
        self.stochastic_seasonal = (stochastic_seasonal,)
        self.stochastic_freq_seasonal = (stochastic_freq_seasonal,)
        self.stochastic_cycle = (stochastic_cycle,)
        self.damped_cycle = (damped_cycle,)
        self.cycle_period_bounds = (cycle_period_bounds,)
        self.mle_regression = (mle_regression,)
        self.use_exact_diffuse = (use_exact_diffuse,)

        super(UnobservedComponents, self).__init__()

    # todo: implement this, mandatory
    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.
            core logic
        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list, np.array or ForecastingHorizon, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)
        Returns
        -------
        self : returns an instance of self.
        """

        # implement here
        # IMPORTANT: avoid side effects to y, X, fh

    # todo: implement this, mandatory
    def _predict(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        """Forecast time series at future horizon.
            core logic
        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        return_pred_int : bool, optional (default=False)
            If True, returns prediction intervals for given alpha values.
        alpha : float or list, optional (default=0.95)
        Returns
        -------
        y_pred : pd.Series
            Point predictions
        y_pred_int : pd.DataFrame - only if return_pred_int=True
            Prediction intervals
        """

        # implement here
        # IMPORTANT: avoid side effects to X, fh
