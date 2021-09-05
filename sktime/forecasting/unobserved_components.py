# -*- coding: utf-8 -*-

from sktime.forecasting.base.adapters import _StatsModelsAdapter


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

        self.level = level
        self.trend = trend
        self.seasonal = seasonal
        self.freq_seasonal = freq_seasonal
        self.cycle = cycle
        self.autoregressive = autoregressive
        self.irregular = irregular
        self.stochastic_level = stochastic_level
        self.stochastic_trend = stochastic_trend
        self.stochastic_seasonal = stochastic_seasonal
        self.stochastic_freq_seasonal = stochastic_freq_seasonal
        self.stochastic_cycle = stochastic_cycle
        self.damped_cycle = damped_cycle
        self.cycle_period_bounds = cycle_period_bounds
        self.mle_regression = mle_regression
        self.use_exact_diffuse = use_exact_diffuse

        super(UnobservedComponents, self).__init__()

    # todo: implement this, mandatory
    def _fit_forecaster(self, y, X=None):
        pass
