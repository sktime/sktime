# -*- coding: utf-8 -*-
import numpy as np

from sktime.transformations.base import _SeriesToSeriesTransformer
from sktime.utils.validation.series import check_series

__author__ = ["Afzal Ansari"]
__all__ = ["CosineTransformer"]


class CosineTransformer(_SeriesToSeriesTransformer):
    """
    Example
    ----------
    >>> from sktime.transformations.series.cos import CosineTransformer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = CosineTransformer()
    >>> y_hat = transformer.fit_transform(y)
    """

    _tags = {"transform-returns-same-time-index": True, "fit-in-transform": True}

    def transform(self, Z, X=None):
        self.check_is_fitted()
        Z = check_series(Z)
        return np.cos(Z)

    # def inverse_transform(self, Z, X=None):
    # only defined for inputs in (-1, 1) range, requires adding extra input check
    # self.check_is_fitted()
    # Z = check_series(Z)
    # return np.arccos(Z)
