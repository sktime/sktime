#!/usr/bin/env python3 -u
# coding: utf-8
<<<<<<< HEAD
=======
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed

__author__ = ["Markus Löning"]
__all__ = []

<<<<<<< HEAD
from sktime.utils.validation.forecasting import check_y, check_sp
from statsmodels.tsa.stattools import acf
from warnings import warn
=======
from warnings import warn

>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed
import numpy as np
from sktime.utils.validation.forecasting import check_sp
from sktime.utils.validation.forecasting import check_y
from statsmodels.tsa.stattools import acf


def autocorrelation_seasonality_test(y, sp):
    """Seasonality test used in M4 competition

    Parameters
    ----------
    sp : int
        Seasonal periodicity

    Returns
    -------
    is_seasonal : bool
        Test result

    References
    ----------
<<<<<<< HEAD
    ..[1]  https://github.com/Mcompetitions/M4-methods/blob/master/Benchmarks%20and%20Evaluation.R
=======
    ..[1]  https://github.com/Mcompetitions/M4-methods/blob/master
    /Benchmarks%20and%20Evaluation.R
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed
    """
    y = check_y(y)
    sp = check_sp(sp)

    y = np.asarray(y)
    n_timepoints = len(y)

    if sp == 1:
        return False

    if n_timepoints < 3 * sp:
<<<<<<< HEAD
        warn("Did not perform seasonality test, as `y`` is too short for the given `sp`, returned: False")
=======
        warn(
            "Did not perform seasonality test, as `y`` is too short for the "
            "given `sp`, returned: False")
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed
        return False

    else:
        coefs = acf(y, nlags=sp, fft=False)  # acf coefficients
        coef = coefs[sp]  # coefficient to check

        tcrit = 1.645  # 90% confidence level
<<<<<<< HEAD
        limits = tcrit / np.sqrt(n_timepoints) * np.sqrt(np.cumsum(np.append(1, 2 * coefs[1:] ** 2)))
=======
        limits = tcrit / np.sqrt(n_timepoints) * np.sqrt(
            np.cumsum(np.append(1, 2 * coefs[1:] ** 2)))
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed
        limit = limits[sp - 1]  #  zero-based indexing
        return np.abs(coef) > limit
