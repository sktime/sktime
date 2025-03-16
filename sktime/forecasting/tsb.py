# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""TSB Forecasting Method."""

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster


class TSB(BaseForecaster):
    r"""Teunter-Syntetos-Babai method for forecasting intermittent time series.

    Implements the method proposed by Teunter, Syntetos, Babai in [1]_.

    The TSB method is a modification of Croston's method to handle intermittent
    time series with very sporadic demand, providing a more accurate estimation
    of the risk of obsolescence. Instead of forecasting the demand interval,
    TSB forecasts demand probability which is updated every period.

    TSB will predict a constant value for all future times, so it essentially
    provides another notion for the average value of a time series.

    Parameters
    ----------
    alpha : float, default = 0.1
        Smoothing parameter for demand size (d)
    beta : float, default = 0.1
        Smoothing parameter for demand occurrence probability (p)

    Examples
    --------
    >>> from sktime.forecasting.tsb import TSB
    >>> from sktime.datasets import load_PBS_dataset
    >>> y = load_PBS_dataset()
    >>> forecaster = TSB(alpha=0.4, beta=0.05)
    >>> forecaster.fit(y)
    TSB(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])

    See Also
    --------
    Croston

    References
    ----------
    .. [1] Teunter, R.H., Syntetos, A.A. and Zied Babai, M. (2011)
    Intermittent Demand: Linking Forecasting to Inventory Obsolescence.
    European Journal of Operational Research, 214, 606-615.

       https://nixtlaverse.nixtla.io/statsforecast/docs/models/tsb.html
       https://juanitorduz.github.io/tsb_numpyro/
    """

    _tags = {
        # packaging info
        # --------------
        "authors": "swetha3456",
        "maintainers": "swetha3456",
        # estimator type
        # --------------
        "requires-fh-in-fit": False,  # is forecasting horizon already required in fit?
    }

    def __init__(self, alpha=0.1, beta=0.1):
        self.alpha = alpha
        self.beta = beta
        self._f = None
        super().__init__()

    def _fit(self, y, X, fh):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored.

        Returns
        -------
        self : returns an instance of self.
        """
        n_timepoints = len(y)  # Historical period: i.e the input array's length
        alpha = self.alpha
        beta = self.beta

        y = y.to_numpy()  # Transform the input into a numpy array
        # Fit the parameters:
        # demand size prediction (d), probability of demand occurrence (p),
        # forecast (f = d * p)
        d, p, f = np.full((3, n_timepoints + 1), np.nan)

        # Initialization:
        first_occurrence = np.argmax(y[:n_timepoints] > 0)
        d[0] = y[first_occurrence]
        p[0] = 0.5
        f[0] = d[0] * p[0]

        # Create t+1 forecasts:
        for t in range(0, n_timepoints):
            if y[t] > 0:
                d[t + 1] = alpha * y[t] + (1 - alpha) * d[t]
                p[t + 1] = beta * 1 + (1 - beta) * p[t]
                f[t + 1] = d[t + 1] * p[t + 1]
            else:
                d[t + 1] = d[t]
                p[t + 1] = (1 - beta) * p[t]
                f[t + 1] = d[t + 1] * p[t + 1]

        self._f = f

        return self

    def _predict(
        self,
        fh=None,
        X=None,
    ):
        """Predict forecast.

        Parameters
        ----------
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored.

        Returns
        -------
        forecast : pd.series
            Predicted forecasts.
        """
        len_fh = len(self.fh)
        f = self._f

        # Predicting future forecasts:to_numpy()
        y_pred = np.full(len_fh, f[-1])

        index = self.fh.to_absolute_index(self.cutoff)
        return pd.Series(y_pred, index=index, name=self._y.name)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict
        """
        params = [
            {},
            {"alpha": 0.5},
            {"alpha": 0, "beta": 0},
            {"alpha": 0.4, "beta": 0.05},
        ]

        return params
