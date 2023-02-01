# -*- coding: utf-8 -*-
# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Croston's Forecasting Method."""


import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster


class Croston(BaseForecaster):
    r"""Croston's method for forecasting intermittent time series.

    Implements the method proposed by Croston in [1]_ and described in [2]_.

    Croston's method is a modification of (vanilla) exponential smoothing to handle
    intermittent time series. A time series is considered intermittent if many
    of its values are zero and the gaps between non-zero entries are not periodic.

    Croston's method will predict a constant value for all future times, so
    Croston's method essentially provides another notion for the average value
    of a time series.

    The method is (equivalent to) the following:

    - Let :math:`v_0,\ldots,v_n` be the non-zero values of the time series
    - Let :math:`v` be the exponentially smoothed average of :math:`v_0,\ldots,v_n`
    - Let :math:`z_0,\ldots,z_n` be the number of consecutive zeros plus 1 between
      the :math:`v_i` in the original time series.
    - Let :math:`z` be the exponentially smoothed average of :math:`z_0,\ldots,z_n`
    - Then the forecast is :math:`\frac{v}{z}`

    The intuition is that :math:`v` is a weighted average of the non-zero time
    series values and :math:`\frac{1}{z}` estimates the probability of getting a
    non-zero value.

    Example to illustrate the :math:`v` and :math:`z` notation.

    - If the original time series is :math:`0,0,2,7,0,0,0,-5` then:

        - The :math:`v`'s are :math:`2,7,-5`
        - The :math:`z`'s are :math:`3,1,4`

    Parameters
    ----------
    smoothing : float, default = 0.1
        Smoothing parameter in exponential smoothing

    Examples
    --------
    >>> from sktime.forecasting.croston import Croston
    >>> from sktime.datasets import load_PBS_dataset
    >>> y = load_PBS_dataset()
    >>> forecaster = Croston(smoothing=0.1)
    >>> forecaster.fit(y)
    Croston(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])

    See Also
    --------
    ExponentialSmoothing

    References
    ----------
    .. [1] J. D. Croston. Forecasting and stock control for intermittent demands.
       Operational Research Quarterly (1970-1977), 23(3):pp. 289–303, 1972.
    .. [2] N. Vandeput. Forecasting Intermittent Demand with the Croston Model.
       https://towardsdatascience.com/croston-forecast-model-for-intermittent-demand-360287a17f5f
    """

    _tags = {
        "requires-fh-in-fit": False,  # is forecasting horizon already required in fit?
    }

    def __init__(self, smoothing=0.1):
        # hyperparameter
        self.smoothing = smoothing
        self._f = None
        super(Croston, self).__init__()

    def _fit(self, y, X=None, fh=None):
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
        smoothing = self.smoothing

        y = y.to_numpy()  # Transform the input into a numpy array
        # Fit the parameters: level(q), periodicity(a) and forecast(f)
        q, a, f = np.full((3, n_timepoints + 1), np.nan)
        p = 1  # periods since last demand observation

        # Initialization:
        first_occurrence = np.argmax(y[:n_timepoints] > 0)
        q[0] = y[first_occurrence]
        a[0] = 1 + first_occurrence
        f[0] = q[0] / a[0]

        # Create t+1 forecasts:
        for t in range(0, n_timepoints):
            if y[t] > 0:
                q[t + 1] = smoothing * y[t] + (1 - smoothing) * q[t]
                a[t + 1] = smoothing * p + (1 - smoothing) * a[t]
                f[t + 1] = q[t + 1] / a[t + 1]
                p = 1
            else:
                q[t + 1] = q[t]
                a[t + 1] = a[t]
                f[t + 1] = f[t]
                p += 1
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

        index = self.fh.to_absolute(self.cutoff)
        return pd.Series(y_pred, index=index, name=self._y.name)

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
        params1 = {}
        params2 = {"smoothing": 0.42}

        return [params1, params2]
