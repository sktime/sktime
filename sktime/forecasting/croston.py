# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.base._base import DEFAULT_ALPHA


class Croston(BaseForecaster):
    """Croston's Forecasting Method.

    This was designed for forecasting intermittent demand.

    Parameters
    -----------
    smoothing : float, default = 0.1
        Smoothing parameter

    Examples
    --------
    >>> from sktime.forecasting.croston import Croston
    >>> from sktime.datasets import load_PBS_dataset
    >>> y = load_PBS_dataset()
    >>> forecaster = Croston(smoothing=0.1)
    >>> forecaster.fit(y)
    Croston(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])

    References
    ----------
    [1]  J. D. Croston. Forecasting and stock control for intermittent demands.
        Operational Research Quarterly (1970-1977), 23(3):pp. 289–303, 1972.
    [2]  Forecasting: Principles and Practice,
        Otext book by Rob J Hyndman and George Athanasopoulos
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
            Exogenous variables are ignored
        Returns
        -------
        self : returns an instance of self.
        """
        if X is not None:
            raise NotImplementedError(
                "Support for exogenous variables is not yet implemented"
            )

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
        return_pred_int=False,
        alpha=DEFAULT_ALPHA,
    ):
        """Predict forecast.

        Parameters
        ----------
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored
        Returns
        -------
        forecast : pd.series
                   predicted forecasts
        """
        if return_pred_int or X is not None:
            raise NotImplementedError()

        len_fh = len(self.fh)
        f = self._f

        # Predicting future forecasts:to_numpy()
        y_pred = np.full(len_fh, f[-1])

        index = self.fh.to_absolute(self.cutoff)
        return pd.Series(y_pred, index=index)
