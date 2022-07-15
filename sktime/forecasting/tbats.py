# -*- coding: utf-8 -*-
# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements TBATS algorithm.

TBATS refers to Exponential smoothing state space model with Trigonometric
Seasonality, Box-Cox transformation, ARMA errors, Trend and Seasonal components.

Wrapping implementation in [1]_ of method proposed in [2]_.
"""

__author__ = ["Martin Walter"]
__all__ = ["TBATS"]

from sktime.forecasting.base.adapters import _TbatsAdapter
from sktime.utils.validation._dependencies import _check_soft_dependencies

_check_soft_dependencies("tbats", severity="warning")


class TBATS(_TbatsAdapter):
    """TBATS forecaster for time series with multiple seasonality.

    Wrapping implementation in [1]_ of method proposed in [2]_. See [3]_
    for blogpost by a creator of [1]_ giving brief explanation of the TBATS
    model. See [4]_ for discussion on multiple seasonalities and discussion of
    how TBATS compares with some other approaches.

    TBATS is acronym for:

    - Trigonometric seasonality
    - Box-Cox transformation
    - ARMA errors
    - Trend
    - Seasonal components

    TBATS was designed to forecast time series with multiple seasonal
    periods. For example, daily data may have a weekly pattern as well
    as an annual pattern. Or hourly data can have three seasonal periods:
    a daily pattern, a weekly pattern, and an annual pattern.

    In TBATS, a Box-Cox transformation is applied to the original time series,
    and then this is modelled as a linear combination of an exponentially
    smoothed trend, a seasonal component and an ARMA component. The seasonal
    components are modelled by trigonometric functions via Fourier series. TBATS
    conducts some hyper-parameter tuning (e.g. which of these components to
    keep and which to discard) using AIC.

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
    sp: Iterable or array-like of floats, optional (default=None)
        Abbreviation of "seasonal periods". The length of each of the periods
        (amount of observations in each period). Accepts int and float values here.
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

    See Also
    --------
    BATS

    References
    ----------
    .. [1] https://github.com/intive-DataScience/tbats
    .. [2] De Livera, A.M., Hyndman, R.J., & Snyder, R. D. (2011),
       Forecasting time series with complex seasonal patterns using exponential
       smoothing, Journal of the American Statistical Association, 106(496), 1513-1527.
       DOI: https://doi.org/10.1198/jasa.2011.tm09771
    .. [3] G. Skorupa. Multiple Seasonalities using TBATS in Python.
       https://medium.com/intive-developers/forecasting-time-series-with-multiple-seasonalities-using-tbats-in-python-398a00ac0e8a
    .. [4] R.J. Hyndman, G. Athanasopoulos. Forecasting: Principles and Practice.
       https://otexts.com/fpp2/complexseasonality.html

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.tbats import TBATS
    >>> y = load_airline()
    >>> forecaster = TBATS(  # doctest: +SKIP
    ...     use_box_cox=False,
    ...     use_trend=False,
    ...     use_damped_trend=False,
    ...     sp=12,
    ...     use_arma_errors=False,
    ...     n_jobs=1)
    >>> forecaster.fit(y)  # doctest: +SKIP
    TBATS(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])  # doctest: +SKIP
    """  # noqa: E501

    _fitted_param_names = "aic"

    def _create_model_class(self):
        """Create model class."""
        # both bats and tbats inherit the same interface from the base class and only
        # instantiate a different model class internally
        from tbats import TBATS as _TBATS

        self._ModelClass = _TBATS

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.


        Returns
        -------
        params : dict or list of dict
        """
        params = {
            "use_box_cox": False,
            "use_trend": False,
            "use_damped_trend": False,
            "sp": [],
            "use_arma_errors": False,
            "n_jobs": 1,
        }
        return params
