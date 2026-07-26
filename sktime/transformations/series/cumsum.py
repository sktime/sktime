# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements a cumulative sum transformation."""

import numpy as np

from sktime.transformations.base import BaseTransformer

__author__ = ["Vaishnav88sk"]
__all__ = ["CumSumTransformer"]


class CumSumTransformer(BaseTransformer):
    """Cumulative sum transformation.

    This is a wrapper around numpy's cumsum function (see :func:`numpy.cumsum`).

    See Also
    --------
    numpy.cumsum

    Examples
    --------
    >>> from sktime.transformations.series.cumsum import CumSumTransformer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = CumSumTransformer()
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
        "capability:inverse_transform": True,
    }

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
        Xt = np.cumsum(X, axis=0)
        return Xt

    def _inverse_transform(self, X, y=None):
        """Inverse transform X and return an inverse transformed version.

        Parameters
        ----------
        X : 2D np.ndarray
            Data to be transformed
        y : ignored argument for interface compatibility

        Returns
        -------
        Xt : 2D np.ndarray
            inverse transformed version of X
        """
        Xt = np.diff(X, axis=0, prepend=0)
        return Xt
