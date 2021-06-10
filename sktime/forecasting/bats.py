#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Martin Walter"]
__all__ = ["BATS"]

from sktime.utils.validation._dependencies import _check_soft_dependencies
from sktime.forecasting.base.adapters import _TbatsAdapter

_check_soft_dependencies("tbats")


class BATS(_TbatsAdapter):
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
    sp: tbats original name of this arg is "seasonal_periods".
        Iterable or array-like of floats, optional (default=None)
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

    Example
    ----------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.bats import BATS
    >>> y = load_airline()
    >>> forecaster = BATS(
    ...     use_box_cox=False,
    ...     use_trend=False,
    ...     use_damped_trend=False,
    ...     sp=12,
    ...     use_arma_errors=False,
    ...     n_jobs=1)
    >>> forecaster.fit(y)
    BATS(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])
    """

    _fitted_param_names = "aic"

    # both bats and tbats inherit the same interface from the base class and only
    # instantiate a different model class internally
    from tbats import BATS as _BATS

    _ModelClass = _BATS
