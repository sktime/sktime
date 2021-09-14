#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements transformers for summarizing a time series."""

__author__ = ["Markus Löning"]
__all__ = ["MeanTransformer"]

import numpy as np

from sktime.transformations.base import _SeriesToPrimitivesTransformer
from sktime.utils.validation.series import check_series


class MeanTransformer(_SeriesToPrimitivesTransformer):
    """Get mean value of time series.

    Examples
    --------
    >>> from sktime.transformations.series.summarize import MeanTransformer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = MeanTransformer()
    >>> y_mean = transformer.fit_transform(y)
    """

    def transform(self, Z, X=None):
        """Transform series.

        Parameters
        ----------
        Z : pd.Series

        Returns
        -------
        float/int
        """
        self.check_is_fitted()
        Z = check_series(Z)
        return np.mean(Z, axis=0)
