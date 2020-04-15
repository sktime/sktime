#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = []

from sktime.utils.validation.forecasting import check_y, check_sp
from statsmodels.tsa.stattools import acf
from warnings import warn
import numpy as np


def autocorrelation_seasonality_test(y, sp):
    """Seasonality test used in M4 competition

    References
    ----------
    ..[1]  https://github.com/Mcompetitions/M4-methods/blob/master/Benchmarks%20and%20Evaluation.R
    """
    y = check_y(y)
    sp = check_sp(sp)

    y = np.asarray(y)
    n_timepoints = len(y)

    if sp == 1:
        return False

    if n_timepoints < 3 * sp:
        warn("Did not perform seasonality test, as `y`` is too short for the given `sp`, returned: False")
        return False

    else:
        coefs = acf(y, nlags=sp, fft=False)  # acf coefficients
        coef = coefs[sp]  # coefficient to check

        tcrit = 1.645  # 90% confidence level
        limits = tcrit / np.sqrt(n_timepoints) * np.sqrt(np.cumsum(np.append(1, 2 * coefs[1:] ** 2)))
        limit = limits[sp - 1]  #  zero-based indexing
        return np.abs(coef) > limit
