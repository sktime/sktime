#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements cosine transformation."""

from math import pi

import numpy as np

from sktime.transformations.base import BaseTransformer

__author__ = ["afzal442"]
__all__ = ["CosineTransformer"]


class CosineTransformer(BaseTransformer):
    """Cosine transformation.

    This is a wrapper around numpy's cosine function (see :func:`numpy.cos`).

    See Also
    --------
    numpy.cos

    Examples
    --------
    >>> from sktime.transformations.series.cos import CosineTransformer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = CosineTransformer()
    >>> y_hat = transformer.fit_transform(y)
    """

    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": "np.ndarray",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "univariate-only": False,
        "fit_is_empty": True,
        "transform-returns-same-time-index": True,
        "capability:inverse_transform": True,
        "capability:inverse_transform:range": [-pi, pi],
    }

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : 2D np.ndarray
            Data to be transformed
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : 2D np.ndarray
            transformed version of X
        """
        Xt = np.cos(X)
        return Xt

    def _inverse_transform(self, X, y=None):
        """Inverse transform X and return an inverse transformed version.

        core logic

        Parameters
        ----------
        X : 2D np.ndarray
            Data to be transformed
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : 2D np.ndarray
            inverse transformed version of X
        """
        Xt = np.arccos(X)
        return Xt
