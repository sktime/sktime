# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements a clip transformation."""

import numpy as np

from sktime.transformations.base import BaseTransformer

__author__ = ["Vaishnav88sk"]
__all__ = ["ClipTransformer"]


class ClipTransformer(BaseTransformer):
    """Clip (limit) the values in an array.

    This is a wrapper around numpy's clip function (see :func:`numpy.clip`).
    Given an interval, values outside the interval are clipped to the interval edges.

    Parameters
    ----------
    a_min : scalar, default=None
        Minimum value. If None, clipping is not performed on lower interval edge.
    a_max : scalar, default=None
        Maximum value. If None, clipping is not performed on upper interval edge.

    See Also
    --------
    numpy.clip

    Examples
    --------
    >>> from sktime.transformations.series.clip import ClipTransformer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = ClipTransformer(a_max=400)
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

    def __init__(self, a_min=None, a_max=None):
        self.a_min = a_min
        self.a_max = a_max
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
        Xt = np.clip(X, a_min=self.a_min, a_max=self.a_max)
        return Xt
