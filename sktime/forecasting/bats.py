#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Martin Walter"]
__all__ = ["BATS"]

import pandas as pd

from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.base._sktime import _SktimeForecaster
from sktime.forecasting.base._sktime import _OptionalForecastingHorizonMixin
from sktime.utils.check_imports import _check_soft_dependencies
from sktime.utils.validation.forecasting import check_y

_check_soft_dependencies("tbats")


class BATS(_OptionalForecastingHorizonMixin, _SktimeForecaster):
    """BATS estimator used to fit and select best performing model.
    BATS (Exponential smoothing state space model with Box-Cox
    transformation, ARMA errors, Trend and Seasonal components.)
    Model has been described in De Livera, Hyndman & Snyder (2011).

    Parameters
    ----------
    use_box_cox: bool or None, optional (default=None)
        If Box-Cox transformation of original series should be applied.
        When None both cases shall be considered and better is selected by AIC.
    box_cox_bounds: tuple, shape=(2,), optional (default=(0, 1))
        Minimal and maximal Box-Cox parameter values.
    use_trend: bool or None, optional (default=None)
        Indicates whether to include a trend or not.
        When None both cases shall be considered and better is selected by AIC.
    use_damped_trend: bool or None, optional (default=None)
        Indicates whether to include a damping parameter in the trend or not.
        Applies only when trend is used.
        When None both cases shall be considered and better is selected by AIC.
    seasonal_periods: iterable or array-like of floats, optional (default=None)
        Length of each of the periods (amount of observations in each period).
        Accepts int and float values here.
        When None or empty array, non-seasonal model shall be fitted.
    use_arma_errors: bool, optional (default=True)
        When True BATS will try to improve the model by modelling residuals with ARMA.
        Best model will be selected by AIC.
        If False, ARMA residuals modeling will not be considered.
    show_warnings: bool, optional (default=True)
        If warnings should be shown or not.
        Also see Model.warnings variable that contains all model related warnings.
    n_jobs: int, optional (default=None)
        How many jobs to run in parallel when fitting BATS model.
        When not provided BATS shall try to utilize all available cpu cores.
    multiprocessing_start_method: str, optional (default='spawn')
        How threads should be started. See also:
        https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
    context: abstract.ContextInterface, optional (default=None)
        For advanced users only. Provide this to override default behaviors
    """

    def __init__(
        self,
        use_box_cox=None,
        box_cox_bounds=(0, 1),
        use_trend=None,
        use_damped_trend=None,
        seasonal_periods=None,
        use_arma_errors=True,
        show_warnings=True,
        n_jobs=None,
        multiprocessing_start_method="spawn",
        context=None,
    ):

        self.use_box_cox = use_box_cox
        self.box_cox_bounds = box_cox_bounds
        self.use_trend = use_trend
        self.use_damped_trend = use_damped_trend
        self.seasonal_periods = seasonal_periods
        self.use_arma_errors = use_arma_errors
        self.show_warnings = show_warnings
        self.n_jobs = n_jobs
        self.multiprocessing_start_method = multiprocessing_start_method
        self.context = context

        super(BATS, self).__init__()

        # import inside method to avoid hard dependency
        from tbats import BATS as _BATS

        self._forecaster = _BATS(
            use_box_cox=use_box_cox,
            box_cox_bounds=box_cox_bounds,
            use_trend=use_trend,
            use_damped_trend=use_damped_trend,
            seasonal_periods=seasonal_periods,
            use_arma_errors=use_arma_errors,
            show_warnings=show_warnings,
            n_jobs=n_jobs,
            multiprocessing_start_method=multiprocessing_start_method,
            context=context,
        )

    def fit(self, y, X=None, fh=None):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables (ignored)

        Returns
        -------
        self : returns an instance of self.
        """
        if X is not None:
            raise NotImplementedError("BATS/TBATS don't support exog or endog data.")

        y = check_y(y)
        self._set_y_X(y, X)
        self._set_fh(fh)

        self._forecaster = self._forecaster.fit(y)
        self._is_fitted = True
        return self

    def _predict(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        index_absolute = None
        if not fh.is_relative:
            index_absolute = fh._values
            fh = fh.to_relative(cutoff=self.cutoff)
        steps = fh.to_pandas().max()
        out = self._forecaster.forecast(steps=steps, confidence_level=1 - alpha)[1]

        # pred
        pred = pd.Series(
            out["mean"][fh.to_indexer(self.cutoff)], index=fh.to_absolute(self.cutoff)
        )

        # pred_int
        upper = pd.Series(
            out["upper_bound"][fh.to_indexer(self.cutoff)],
            index=fh.to_absolute(self.cutoff),
        )
        lower = pd.Series(
            out["lower_bound"][fh.to_indexer(self.cutoff)],
            index=fh.to_absolute(self.cutoff),
        )
        pred_int = pd.DataFrame({"upper": upper, "lower": lower})

        if index_absolute is not None:
            pred.index = index_absolute
            pred_int.index = index_absolute

        if return_pred_int:
            return pred, pred_int
        else:
            return pred
