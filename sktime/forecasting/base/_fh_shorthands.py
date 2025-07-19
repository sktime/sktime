# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Shorthands for defining forecasting horizon."""

__all__ = ["upto"]


def upto(n):
    """Return an integer forecasting horizon that goes up to n, starting from 1.

    Same as range(1, n + 1), i.e., predicting the next n time points or periods.

    Parameters
    ----------
    n : int
        The maximum time point in the forecasting horizon.

    Returns
    -------
    fh : ForecastingHorizon
        A forecasting horizon that goes up to n.
    """
    from sktime.forecasting.base._fh import ForecastingHorizon

    return ForecastingHorizon(range(1, n + 1), is_relative=True)
