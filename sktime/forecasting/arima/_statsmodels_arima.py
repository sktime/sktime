#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Interface to ARIMA and SARIMAX models from statsmodels package."""

__author__ = ["AryanDhanuka10", "fkiraly"]
__all__ = ["StatsmodelsARIMA"]

from sktime.forecasting.base.adapters._statsmodels_arima import _StatsmodelsArimaAdapter


class StatsmodelsARIMA(_StatsmodelsArimaAdapter):
    """ARIMA/SARIMAX forecaster based on statsmodels.

    This class wraps the `statsmodels.tsa.statespace.SARIMAX` model to make it
    compatible with sktime's forecasting API.

    Parameters
    ----------
    order : tuple of int, default=(1, 0, 0)
        The (p, d, q) order of the model for the AR, differencing, and MA terms.
    seasonal_order : tuple of int, default=(0, 0, 0, 0)
        The (P, D, Q, s) order of the seasonal component.
    """

    pass
