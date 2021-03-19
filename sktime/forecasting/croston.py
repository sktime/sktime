# -*- coding: utf-8 -*-
"""
Implementation of Croston's Method
----------------------------------
Useful for Forecasting Intermittent Demand Time Series.
The Croston() function produces forecasts using Croston’s method.
It simply uses α = 0.1 by default,
and p = 0 is set to be equal to the first observation in each of the series.
This is consistent with the way Croston envisaged the method being used.

Parameters:
-----------
    demand: array-like
        Historical data
    future_periods: int
        Time period for which predictions are required
    alpha: float, optional(default=0.1)
        Smoothing parameter

Returns:
--------
    forecast: array-like
        Forecasted demand (on average per period) diff
"""

import numpy as np


def Croston(demand, future_periods=1, alpha=0.1):

    d = np.array(demand)  # Transform the input into a numpy array
    cols = len(d)  # Historical period: i.e the demand array's length
    d = np.append(
        d, [np.nan] * future_periods
    )  # Append np.nan into the demand array to cover future periods

    # level(a), periodicity(p) and forecast(f)
    q, a, f = np.full((3, cols + future_periods), np.nan)
    p = 1  # periods since last demand observation

    # Initialization:
    first_occurrence = np.argmax(d[:cols] > 0)
    q[0] = d[first_occurrence]
    a[0] = 1 + first_occurrence
    f[0] = q[0] / a[0]

    # Create t+1 forecasts:
    for t in range(0, cols):
        if d[t] > 0:
            q[t + 1] = alpha * d[t] + (1 - alpha) * q[t]
            a[t + 1] = alpha * p + (1 - alpha) * a[t]
            f[t + 1] = q[t + 1] / a[t + 1]
            p = 1
        else:
            q[t + 1] = q[t]
            a[t + 1] = a[t]
            f[t + 1] = f[t]
            p += 1

    # Future forecasts:
    q[cols + 1 : cols + future_periods] = q[cols]
    a[cols + 1 : cols + future_periods] = a[cols]
    f[cols + 1 : cols + future_periods] = f[cols]

    return np.array(f[cols:])
