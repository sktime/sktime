# -*- coding: utf-8 -*-
# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Wraps the UnobservedComponents (state space) model from statsmodels."""

__all__ = ["UnobservedComponents"]
__author__ = ["Juan Orduz"]

from sktime.forecasting.base.adapters import _StatsModelsAdapter
from statsmodels.tsa.statespace.structural import (
    UnobservedComponents as _UnobservedComponents,
)


class UnobservedComponents(_StatsModelsAdapter):
    """TODO: Add docstrings.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.structural import UnobservedComponents
    >>> y = load_airline()
    >>> forecaster = UnobservedComponents(level='local linear trend')
    >>> forecaster.fit(y)
    UnobservedComponents(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])
    """

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
        start_params=None,
        transformed=True,
        includes_fixed=False,
        cov_type=None,
        cov_kwds=None,
        method="lbfgs",
        maxiter=50,
        full_output=1,
        disp=5,
        callback=None,
        return_params=False,
        optim_score=None,
        optim_complex_step=None,
        optim_hessian=None,
        flags=None,
        low_memory=False,
        **kwargs
    ):
        # Model params
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

        # Fit params
        self.start_params = start_params
        self.transformed = transformed
        self.includes_fixed = includes_fixed
        self.cov_type = cov_type
        self.cov_kwds = cov_kwds
        self.method = method
        self.maxiter = maxiter
        self.full_output = full_output
        self.disp = disp
        self.callback = callback
        self.return_params = return_params
        self.optim_score = optim_score
        self.optim_complex_step = optim_complex_step
        self.optim_hessian = optim_hessian
        self.flags = flags
        self.low_memory = low_memory

        super(UnobservedComponents, self).__init__()

    def _fit_forecaster(self, y, X=None):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored
        """
        self._forecaster = _UnobservedComponents(
            endog=y,
            exog=X,
            level=self.level,
            trend=self.trend,
            seasonal=self.seasonal,
            freq_seasonal=self.freq_seasonal,
            cycle=self.cycle,
            autoregressive=self.autoregressive,
            irregular=self.irregular,
            stochastic_level=self.stochastic_level,
            stochastic_trend=self.stochastic_trend,
            stochastic_seasonal=self.stochastic_seasonal,
            stochastic_freq_seasonal=self.stochastic_freq_seasonal,
            stochastic_cycle=self.stochastic_cycle,
            damped_cycle=self.damped_cycle,
            cycle_period_bounds=self.cycle_period_bounds,
            mle_regression=self.mle_regression,
            use_exact_diffuse=self.use_exact_diffuse,
        )

        self._fitted_forecaster = self._forecaster.fit(
            start_params=self.start_params,
            transformed=self.transformed,
            includes_fixed=self.includes_fixed,
            cov_type=self.cov_type,
            cov_kwds=self.cov_kwds,
            method=self.method,
            maxiter=self.maxiter,
            full_output=self.full_output,
            disp=self.disp,
            callback=self.callback,
            return_params=self.return_params,
            optim_score=self.optim_score,
            optim_complex_step=self.optim_complex_step,
            optim_hessian=self.optim_hessian,
            flags=self.flags,
            low_memory=self.low_memory,
        )

    def summary(self):
        """Get a summary of the fitted forecaster.

        This is the same as the implementation in statsmodels:
        https://www.statsmodels.org/dev/examples/notebooks/generated/statespace_structural_harvey_jaeger.html
        """
        return self._fitted_forecaster.summary()

    def simulate(
        self,
        nsimulations,
        X=None,
        measurement_shocks=None,
        state_shocks=None,
        initial_state=None,
        anchor=None,
        repetitions=None,
        **kwargs
    ):
        """TODO: Add docstrings."""
        return self._fitted_forecaster.simulate(
            nsimulations=nsimulations,
            measurement_shocks=measurement_shocks,
            state_shocks=state_shocks,
            initial_state=initial_state,
            anchor=anchor,
            repetitions=repetitions,
            exog=X,
            **kwargs
        )

    def plot_diagnostics(
        self, variable=0, lags=10, fig=None, figsize=None, truncate_endog_names=24
    ):
        """TODO: Add docstrings."""
        self._fitted_forecaster.plot_diagnostics(
            variable=variable,
            lags=lags,
            fig=fig,
            figsize=figsize,
            truncate_endog_names=truncate_endog_names,
        )

    # TODO: This plot function generates an error:
    # "TypeError: float() argument must be a string or a number, not 'Period'"
    #
    # def plot_components(
    #     self,
    #     which=None,
    #     alpha=0.05,
    #     observed=True,
    #     level=True,
    #     trend=True,
    #     seasonal=True,
    #     freq_seasonal=True,
    #     cycle=True,
    #     autoregressive=True,
    #     legend_loc="upper right",
    #     fig=None,
    #     figsize=None,
    # ):
    #     """TODO: Add docstrings."""
    #     self._fitted_forecaster.plot_components(
    #         which=which,
    #         alpha=alpha,
    #         observed=observed,
    #         level=level,
    #         trend=trend,
    #         seasonal=seasonal,
    #         freq_seasonal=freq_seasonal,
    #         cycle=cycle,
    #         autoregressive=autoregressive,
    #         legend_loc=legend_loc,
    #         fig=fig,
    #         figsize=figsize,
    #     )
