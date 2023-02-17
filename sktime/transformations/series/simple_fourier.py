# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

"""Simple Fourier transform for time series, interfacing numpy rfft."""

__author__ = ["blazingbhavneek"]


import numpy as np
import pandas as pd
from numpy.fft import rfft

from sktime.transformations.base import BaseTransformer


class SimpleFourier(BaseTransformer):
    r"""Simple Fourier transform for time series.

    Fourier Transform is used to test data for periodicity.
    Non-periodic/Noisy/Random data will have limited and
    random amplitudes for frequencies, whereas periodic data
    will have exceptional amplitudes for certain frequencies.

    The implementation is based on the fast fourier transform from numpy.fft

    Examples
    --------
    >>> from sktime.transformations.series.simple_fourier import SimpleFourier
    >>> from sktime.datasets import load_airline
    >>> X = load_airline()
    >>> transformer = SimpleFourier()
    >>> X_ft = transformer.fit_transform(X)
    """

    _tags = {
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:instancewise": True,
        "scitype:transform-labels": "None",
        "X_inner_mtype": "pd.Series",
        "y_inner_mtype": "None",
        "univariate-only": False,
        "requires_y": False,
        "fit_is_empty": True,
        "capability:inverse_transform": False,
        "capability:unequal_length": True,
        "handles-missing-data": False,
    }

    def __init__(self):
        super(SimpleFourier, self).__init__()

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : Series mtype X_inner_mtype

        Returns
        -------
        transformed version of X
        """
        # numpy.fft methods
        dft_seq = np.abs(rfft(X))

        # Combining the arrays to Pandas Series
        Y = pd.Series(dft_seq[1:])
        return Y
