# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements a round transformation."""

import numpy as np

from sktime.transformations.base import BaseTransformer

__author__ = ["Vaishnav88sk"]
__all__ = ["RoundTransformer"]


class RoundTransformer(BaseTransformer):
    """Round transformation.

    This is a wrapper around numpy's round function (see :func:`numpy.round`).
    It rounds an array to the given number of decimals.

    Parameters
    ----------
    decimals : int, default=0
        Number of decimal places to round to.

    See Also
    --------
    numpy.round

    Examples
    --------
    >>> from sktime.transformations.series.round import RoundTransformer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = RoundTransformer(decimals=1)
    >>> y_hat = transformer.fit_transform(y)
    """

    _tags = {
        "authors": "Vaishnav88sk",
        "maintainers": "Vaishnav88sk",
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:instancewise": True,
        "X_inner_mtype": "np.ndarray",
        "y_inner_mtype": "None",
        "capability:multivariate": True,
        "fit_is_empty": True,
        "transform-returns-same-time-index": True,
        "capability:inverse_transform": False,
    }

    def __init__(self, decimals=0):
        self.decimals = decimals
        super().__init__()

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        Parameters
        ----------
        X : 2D np.ndarray
            Data to be transformed
        y : ignored argument for interface compatibility

        Returns
        -------
        Xt : 2D np.ndarray
            transformed version of X
        """
        Xt = np.round(X, decimals=self.decimals)
        return Xt
